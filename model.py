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