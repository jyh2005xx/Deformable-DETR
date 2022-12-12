# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, non_obj_class=False, num_bbox_dim=4):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.non_obj_class = non_obj_class
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_bbox_dim, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # import IPython
        # IPython.embed()
        # assert(0)
        
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    def forward(self, inputs):
        out = torch.cat([torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, non_obj_class= False, bbox_embedding='grid'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.num_projection = 5000
        self.rand_bbox = True
        self.non_obj_class = non_obj_class
        self.bbox_embedding = bbox_embedding
        if self.bbox_embedding == 'pos_encoding' or self.bbox_embedding == 'pos_decoding':
            self.pos_encoder = NerfPositionalEncoding(10)
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def box2grid(self,boxes):
        B,_ = boxes.shape
        num_pt = 6
        boxes_x0 = boxes[:,[0]] - boxes[:,[2]]/2
        boxes_y0 = boxes[:,[1]] - boxes[:,[3]]/2
        interval = torch.arange(0,num_pt,device=boxes.device) / (num_pt-1)
        boxes_x = boxes_x0 + interval* boxes[:,[2]]
        boxes_y = boxes_y0 + interval* boxes[:,[3]]
        grid = torch.stack([boxes_x.unsqueeze(-1).repeat(1,1,num_pt),boxes_y.unsqueeze(-2).repeat(1,num_pt,1)],dim=-1).view(B,-1)
        return grid

    def loss_swas(self,outputs,targets, indices, num_boxes, log=True):
        # prepare input to sw loss
        # N by M by 4+num_classess

        B, N, _ = outputs['pred_boxes'].shape
        
        # prepare bbox for sw loss
        if self.bbox_embedding == 'pos_encoding':
            pre_grid = self.pos_encoder(outputs['pred_boxes']).view(B,N,-1)
        elif self.bbox_embedding == 'pos_decoding':
            pre_grid = outputs['pred_boxes']
        elif self.bbox_embedding == 'grid':
            pre_grid = self.box2grid(outputs['pred_boxes'].view(-1,4)).view(B,N,-1)
        
        _, _, N_B = pre_grid.shape
        
        # prepare class prob for sw loss
        if not self.non_obj_class:
            pred_logits = (outputs['pred_logits']).sigmoid()
        else:
            pred_logits = nn.Softmax(dim=2)(outputs['pred_logits'])

        # concat bbox and class prob
        pre_sw = torch.cat((pre_grid,pred_logits),2)
        
        # prepare ground truth
        with torch.no_grad():
            # generate random bbox
            gt_box = torch.zeros(B,N,4,device = pre_sw.device)
            if self.rand_bbox:
                # bbox is in xywh format, value is between 0 and 1 
                # random x and h between 0.1 and 0.7
                gt_box[:,:,2:4] = torch.rand(pre_sw.shape[0],pre_sw.shape[1],2,device=pre_sw.device)*0.6+0.1
                # random center to make sure bbox with in image
                gt_box[:,:,:2] = (1-gt_box[:,:,2:4])*torch.rand(pre_sw.shape[0],pre_sw.shape[1],2,device=pre_sw.device)+gt_box[:,:,2:4]/2
                # !!!!!!!!!!!!!!!! change to corner and range to -0.5,+0.5
                # gt_box[:,:,:4] = box_ops.box_cxcywh_to_xyxy(gt_box[:,:,:4]) - 0.5
            else:
                gt_box[:,:,:2] = 0
                gt_box[:,:,:2] = 0.5
                # !!!!!!!!!!!!!!!! change to corner and range to -0.5,+0.5
                # gt_box[:,:,:4] = 0

            # generate class prob holder
            gt_cls = torch.zeros(outputs['pred_logits'].shape,device = pre_sw.device)
            if self.non_obj_class:
                gt_cls[:,:,-1]=1   

            # fill in gt        
            for idx in range(len(targets)):
                _n = targets[idx]['labels'].shape[0]
                gt_box[idx,:_n,:4] = targets[idx]['boxes']
                gt_cls[idx,:_n] = F.one_hot(targets[idx]['labels'],self.num_classes)
            
            # reparameterize gt box
            if self.bbox_embedding == 'pos_encoding':
                gt_grid = self.pos_encoder.forward(gt_box).view(B,N,-1)
            elif self.bbox_embedding == 'pos_decoding':
                gt_grid = self.pos_encoder.forward(gt_box-0.5).view(B,N,-1)
            elif self.bbox_embedding == 'grid':
                gt_grid = self.box2grid(gt_box.view(-1,4)).view(B,N,-1)
            else:
                print('non defined bbox embedding')
                assert(0)
            
            # concat bbox and class
            gt_sw = torch.cat([gt_grid,gt_cls],dim=-1)

        with torch.no_grad():
            lose_sw_bbox = self.sw_loss(pre_sw[:,:,:N_B],gt_sw[:,:,:N_B])
            lose_sw_class = self.sw_loss(pre_sw[:,:,N_B:],gt_sw[:,:,N_B:])
        if True:
            if self.bbox_embedding == 'grid':
                lose_sw = self.sw_loss(pre_sw-0.5,gt_sw-0.5)
            else:
                pre_sw[:,:,N_B:] = pre_sw[:,:,N_B:] -0.5
                gt_sw[:,:,N_B:] = gt_sw[:,:,N_B:] -0.5
                pre_sw[:,:,:N_B] = pre_sw[:,:,:N_B] /2
                gt_sw[:,:,:N_B] = gt_sw[:,:,:N_B] /2
                lose_sw = self.sw_loss(pre_sw,gt_sw)
            # print('--------Predict---------')
            # print(pre_sw[0,:10,:20])
            # print('-----------GT-----------')
            # print(gt_sw[0,:10,:20])
        else:
            lose_sw = self.sw_loss(pre_sw[:,:,:N_B],gt_sw[:,:,:N_B])
        losses = {'loss_swas': lose_sw,'lose_sw_bbox': lose_sw_bbox,'lose_sw_class': lose_sw_class, 'class_error':torch.zeros(1,device=lose_sw.device)-1}
        return losses

    def sw_loss(self, dist_1,dist_2):
        # normlize
        dist_1 = F.normalize(dist_1, dim=-1, p=2)
        dist_2 = F.normalize(dist_2, dim=-1, p=2)
        latent_channel = dist_1.shape[-1]
        # generate random projection
        theta = torch.nn.functional.normalize(torch.randn((latent_channel, self.num_projection),device=dist_1.device,requires_grad=False), dim=0, p=2)
        # project to 1D
        dist_1 = dist_1@theta
        dist_2 = dist_2@theta
        # sort
        dist_1, _ = torch.sort(dist_1,1)
        dist_2, _ = torch.sort(dist_2,1)
        # L2 wasserstein distance
        w_dist = torch.mean(torch.sum(torch.pow((dist_1-dist_2),2),dim=1),dim=-1)
        # average distance
        errG = torch.mean(w_dist,dim=0,keepdim=True)
        return errG

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
             'swas':self.loss_swas
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if 'swas' in self.losses:
            indices = None
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'swas' in self.losses:
                    indices = None
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            
            if 'swas' in self.losses:
                indices = None
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, non_obj_class, bbox_embedding ):
        super().__init__()
        self.non_obj_class = non_obj_class
        self.bbox_embedding = bbox_embedding

    @torch.no_grad()
    def forward(self, outputs, target_sizes, debug=False, raw_output=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        # !!!!!!!!!!!!!!!! -0.5,+0.5
        # out_bbox[:,:,:2] = out_bbox[:,:,:2] + 0.5
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        if self.non_obj_class:
            out_logits = nn.Softmax(dim=3)(out_logits)
        else:
            out_logits = out_logits.sigmoid()
        if self.bbox_embedding == 'pos_decoding':
            B,K,D = out_bbox.shape
            xs = out_bbox[:,:,:D//4]
            ys = out_bbox[:,:,D//4:D//2]
            ws = out_bbox[:,:,D//2:D//4*3]
            hs = out_bbox[:,:,D//4*3:]
            out_bbox = torch.zeros(B,K,4,device=out_bbox.device)
            out_bbox[:,:,0] = torch.asin(xs[:,:,0])+0.5
            out_bbox[:,:,1] = torch.asin(ys[:,:,0])+0.5
            out_bbox[:,:,2] = torch.asin(ws[:,:,0])+0.5
            out_bbox[:,:,3] = torch.asin(hs[:,:,0])+0.5
        topk_values, topk_indexes = torch.topk(out_logits.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if not raw_output:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'raw_bbox': raw_b, 'raw_prob': raw_p} for s, l, b, raw_b, raw_p in zip(scores, labels, boxes, out_bbox, out_logits)]

        if debug:
            import IPython
            IPython.embed()
            assert(0)
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    if args.non_obj_class:
        num_classes = num_classes + 1
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        non_obj_class = args.non_obj_class,
        num_bbox_dim = args.num_bbox_dim
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    if args.swas:
        weight_dict = {'loss_swas': 1}
    else:
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    if args.swas:
        losses=['swas']
    else:
        losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,non_obj_class = args.non_obj_class, bbox_embedding = args.bbox_embedding)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args.non_obj_class, args.bbox_embedding)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    return model, criterion, postprocessors
