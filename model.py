import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import *
from torchvision.ops import generalized_box_iou_loss

# -------------------- Models -----------------------

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:7]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        
    def forward(self, img_data):
        return self.backbone(img_data)

class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=30, p_dropout=0.3):
        # print(f"Proposal Module:\nNo. Anchors: {n_anchors}, In Features: {in_features}, Hidden Features: {hidden_dim}")
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)
        
    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # determine mode
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        feature_map = feature_map.to(config.DEVICE)
        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        # print(f"Forward PM\nIn: {feature_map.shape}, out: {out.shape}")
        
        reg_offsets_pred = self.reg_head(out)# (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out) # (B, A, hmap, wmap)
        offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)
        # print(f"Reg offsets: {reg_offsets_pred[0]}, Conf Scores: {conf_scores_pred[0]}, Offsets Pos: {offsets_pos}")
        # print(f"Pos Anc Coords :{pos_anc_coords}")
        
        if mode == 'train': 
            # get conf scores
            # print(f"Pos Anc Indices: {pos_anc_ind.device}, Neg Anc Indices: {neg_anc_ind.device}")
            # print(f"conf_scores_pred: {conf_scores_pred.shape}, pos_anc_indices: {pos_anc_ind}, neg_anc_indices: {neg_anc_ind}")
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # print(conf_scores_neg, conf_scores_pos)
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            # generate proposals using offsets
            # print(f"Offsets: {offsets_pos}")
            proposals = generate_proposals(pos_anc_coords, offsets_pos)
            # print(f"Proposals after generation TRAIN: {proposals}")
            
            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals
            
        elif mode == 'eval':
            # print(f"Conf Scores Pred: {conf_scores_pred}, Reg Offsets Pred: {reg_offsets_pred}")
            return conf_scores_pred, offsets_pos
        
class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        
        # downsampling scale factor 
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        
        # scales and ratios for anchor boxes
        self.anc_scales = [0.25, 0.5, 0.75, 1, 2]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios) * 2
        
        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        
        # weights for loss
        self.w_conf = 1
        self.w_reg = 5
        
        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes)
        
    def forward(self, images, gt_bboxes, gt_classes):
        # print(f"Images: {images.device}, Bboxes: {gt_bboxes.device}, Classes: {gt_classes.device}")
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images).to(config.DEVICE)
        
        # generate anchors
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        # print(f"Anc centers x: {anc_pts_x.shape}, Anc centers Y: {anc_pts_y.shape}")
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        # print(f"Anc Base: {anc_base.shape}")
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        # print(f"Anc Boxes All: {anc_boxes_all.shape}")
        
        # get positive and negative anchors amongst other things
        # print(f"GT Bboxes: {gt_bboxes[0]}")
        gt_bboxes_proj = project_bboxes(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')
        # print(f"GT Bboxes Proj: {gt_bboxes_proj[0]}")
        # print(f"Anc Boxes: {anc_boxes_all.device}, Bboxes Projected: {gt_bboxes_proj.device}, Classes: {gt_classes.device}") 

        positive_anc_ind, negative_anc_ind, GT_conf_scores, \
        GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes, self.pos_thresh, self.neg_thresh)
        
        # pass through the proposal module
        # print(f"Positive Anc Ind: {positive_anc_ind}, Positive Anc Coords Length: {len(positive_anc_coords)}")
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module.forward(feature_map, positive_anc_ind, \
                                                                                        negative_anc_ind, positive_anc_coords)
        
        # print(f"GT Offsets: {GT_offsets}, Offset Pos: {offsets_pos}")
        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        # print(f"CLS Loss RPN: {cls_loss}, REG Loss RPN: {reg_loss}")
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        
        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos

    def inference(self, images, conf_thresh=0.5, nms_thresh=0):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            # generate anchors
            anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
            # print(f"anc pts x: {anc_pts_x.shape}, anc pts y: {anc_pts_y.shape}")
            anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w)).to(config.DEVICE)
            # print(f"anc base shape: {anc_base.shape}")
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            # print(f"Anc boxes example before generate proposals method: {anc_boxes_all[0][-7:-1]}") NO PROBLEM WITH ANC 0<x<+12
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            # get conf scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module.forward(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)
            # print(f"Anc Boxes Flat: {anc_boxes_flat.shape}, Offsets Pred RPN: {offsets_pred.shape}")

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                # print(conf_thresh, conf_scores)
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = generate_proposals(anc_boxes, offsets)
                # print(f"Proposals after generation: {proposals}")
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # print(f"Proposals Pos: {proposals_pos.shape}")
                # filter based on nms threshold
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            # print(f"Proposals Final: {proposals_final[-7:-1]}")
        return proposals_final, conf_scores_final, feature_map
    