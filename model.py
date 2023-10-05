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

class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()        
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        
        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, feature_map, proposals_list, gt_classes=None):
        
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        # apply roi pooling on proposals followed by avg pooling
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        
        # flatten the output
        roi_out = roi_out.squeeze(-1).squeeze(-1).to(config.DEVICE)
        
        # pass the output through the hidden network
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        
        # get the classification scores
        cls_scores = self.cls_head(out)

        # convert scores into probability
        cls_probs = F.softmax(cls_scores, dim=-1)
        # get classes with highest probability
        classes_all = torch.argmax(cls_probs, dim=-1)
        
        if mode == 'eval':
            return cls_scores
        
        # compute cross entropy loss
        # print(f"Cls Scores: {cls_scores}, CLS Probs: {cls_probs}, CLS all: {classes_all}, GT Classes Long: {gt_classes.long()}")
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        
        return cls_loss, classes_all
    
class TwoStageDetector(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__() 
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels).to(device=config.DEVICE)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size).to(device=config.DEVICE)
        
    def forward(self, images, gt_bboxes, gt_classes):
        # print(f"Images: {images.device}, gt_bboxes: {gt_bboxes.device}, gt_classes: {gt_classes.device}")
        # print(f"TWO stage detector gt classes: {gt_classes}")
        total_rpn_loss, feature_map, proposals, \
        positive_anc_ind_sep, GT_class_pos = self.rpn.forward(images, gt_bboxes, gt_classes)
        # print(f"gt_class: {gt_classes}, gt_classes_pos: {GT_class_pos}")
        # print(f"All proposals {proposals.shape}")
        # get separate proposals for each sample
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)
        
        # print(f"Length of Positive Proposals List: {len(pos_proposals_list[0]), len(pos_proposals_list[1]), len(pos_proposals_list[2])}")
        cls_loss, classes_all = self.classifier.forward(feature_map, pos_proposals_list, GT_class_pos)
        # display_inference(images, pos_proposals_list, classes_all, gt_bboxes, gt_classes, config.WIDTH_SCALE_FACTOR, config.HEIGHT_SCALE_FACTOR)
        # print(f"total_rpn_loss: {total_rpn_loss}, cls_loss: {cls_loss}")
        total_loss = cls_loss + total_rpn_loss
        
        return total_loss
    
    def inference(self, images, conf_thresh=0.7, nms_thresh=0.5):
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        print(f"No. Of Proposals after RPN: {len(proposals_final[0])}")
        # print(f"Example: {proposals_final[0][-7:-1]}")
        cls_scores = self.classifier.forward(feature_map, proposals_final)
        
        # convert scores into probability
        cls_probs = F.softmax(cls_scores, dim=-1)
        # get classes with highest probability
        classes_all = torch.argmax(cls_probs, dim=-1)
        
        classes_final = []
        # slice classes to map to their corresponding image
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i]) # get the number of proposals for each image
            classes_final.append(classes_all[c: c+n_proposals])
            c += n_proposals
        
        # print(f"Proposals Shape: {len(proposals_final)}, Classes Shape: {len(classes_final), len(classes_final[0]), classes_final[0]}")
        return proposals_final, conf_scores_final, classes_final, cls_scores

# ------------------- Loss Utils ----------------------

def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_scores_neg)
    
    target = torch.cat((target_pos, target_neg))
    inputs = torch.cat((conf_scores_pos, conf_scores_neg))
     
    loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size
    
    return loss

def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    assert gt_offsets.size() == reg_offsets_pos.size()
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size
    return loss