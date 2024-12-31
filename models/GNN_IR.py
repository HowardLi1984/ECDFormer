from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from utils.eval_func import MAE, MAPE, RMSE_Peak, Accuracy
# from utils.loss.dilate_loss import dilate_loss
from utils.loss.soft_dtw_cuda import SoftDTW
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
from tqdm import tqdm
import json
import numpy as np

class AtomEncoder(torch.nn.Module):

    def __init__(self, args, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        
        for i, dim in enumerate(args.full_atom_feature_dims):
            if i == 1 or i == 5: dim = 9
            emb = torch.nn.Embedding(dim + 5, emb_dim)  # 不同维度的属性用不同的Embedding方法
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class BondEncoder(torch.nn.Module):

    def __init__(self, args, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(args.full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = centers.reshape([1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))

class BondFloatRBF(torch.nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, args, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
        self.bond_float_names = args.bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (nn.Parameter(torch.arange(0, 2, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                # # (centers, gamma)
                # 'prop': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                # 'diameter': (nn.Parameter(torch.arange(3, 12, 0.3)), nn.Parameter(torch.Tensor([1.0]))),
                # ##=========Only for pure GNN===============
                # 'column_TPSA': (nn.Parameter(torch.arange(0, 1, 0.05).to(torch.float32)), nn.Parameter(torch.Tensor([1.0]))),
                # 'column_RASA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                # 'column_RPSA': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                # 'column_MDEC': (nn.Parameter(torch.arange(0, 10, 0.5)), nn.Parameter(torch.Tensor([2.0]))),
                # 'column_MATS': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(self.device), gamma.to(self.device))
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim).cuda()
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class BondAngleFloatRBF(torch.nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, args, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
        self.bond_angle_float_names = args.bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (nn.Parameter(torch.arange(0, torch.pi, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers.to(self.device), gamma.to(self.device))
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape(-1, 1)
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# GNN to generate node embedding
class GINNodeEmbedding(torch.nn.Module):
    """
    Output: node representations
    """
    def __init__(self, args, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module 采用多层GINConv实现图上结点的嵌入。
        """
        super(GINNodeEmbedding, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(args, emb_dim)
        self.bond_encoder = BondEncoder(args, emb_dim)
        self.bond_float_encoder = BondFloatRBF(args, emb_dim)
        self.bond_angle_encoder = BondAngleFloatRBF(args, emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle=torch.nn.ModuleList()
        self.convs_bond_float=torch.nn.ModuleList()
        self.convs_bond_embeding=torch.nn.ModuleList()
        self.convs_angle_float=torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(BondEncoder(args, emb_dim))
            self.convs_bond_float.append(BondFloatRBF(args, emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(args, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_atom_bond, batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba, edge_attr_ba= batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        # computing input node embedding
        h_list = [self.atom_encoder(x)]

        if self.args.Use_geometry_enhanced == True: # 默认进入这一分支
            bond_id_names = self.args.bond_id_names
            h_list_ba=[self.bond_float_encoder(edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))+self.bond_encoder(edge_attr[:,0:len(bond_id_names)].to(torch.int64))]
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
                cur_h_ba = self.convs_bond_embeding[layer](edge_attr[:,0:len(bond_id_names)].to(torch.int64))+self.convs_bond_float[layer](edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))
                cur_angle_hidden = self.convs_angle_float[layer](edge_attr_ba)
                h_ba = self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)
                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                    h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                    h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
                if self.residual:
                    h += h_list[layer]
                    h_ba += h_list_ba[layer]

                h_list.append(h)
                h_list_ba.append(h_ba)

            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
                edge_representation = h_list_ba[-1]
            elif self.JK == "sum":
                node_representation = 0
                edge_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]
                    edge_representation += h_list_ba[layer]

            return node_representation, edge_representation

        if self.args.Use_geometry_enhanced == False:
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index,
                                      self.convs_bond_embeding[layer](edge_attr[:, 0:len(bond_id_names)].to(torch.int64)) +
                                      self.convs_bond_float[layer](edge_attr[:, len(bond_id_names):edge_attr.shape[1] + 1].to(torch.float32)))
                h = self.batch_norms[layer](h)
                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                if self.residual:
                    h += h_list[layer]

                h_list.append(h)

            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "sum":
                node_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]

            return node_representation

def index_transform(raw_index, batch_size, max_node_num):
    # 将原始的 index 分解为可以处理的 index
    # 1. 输出为二维的列表, 元素不定长
    # 2. 输出padding_tensor的位置, 

    def get_index1(lst=None, batch_num=-1):
        return [index for (index,value) in enumerate(lst) if value == batch_num]
    
    raw_index = raw_index.tolist()
    index_list = []
    for batch_id in range(batch_size):
        index_list.append(get_index1(raw_index, batch_id))
    return index_list

def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == -1] = -torch.inf
    return key_padding_mask

def feat_padding_mask(index, max_node_num):
    # 通过 index 将 padding位对应的Mask生成出来
    new_index = []
    for itm_list in index:
        new_index.append(itm_list + [-1]*(max_node_num-len(itm_list)))
    new_index = torch.tensor(new_index)
    return get_key_padding_mask(new_index)

class GNNIR_Model(nn.Module):
    def __init__(self, args, num_tasks=9, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="attention",
                 descriptor_dim=1781):
        """GIN Graph Pooling Module
        此模块首先采用GINNodeEmbedding模块对图上每一个节点做嵌入
        然后对节点嵌入做池化得到图的嵌入
        最后用一层线性变换得到图的最终的表示(graph representation)
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GNNIR_Model, self).__init__()

        self.args = args
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim = descriptor_dim
        self.max_node_num = 63 # 63个单独的Node特征 + 一个整体平均特征 = 64
        
        self.query_embed = nn.Embedding(15, self.emb_dim)
        self.gnn_node = GINNodeEmbedding(args, num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":    self.pool = global_add_pool
        elif graph_pooling == "mean": self.pool = global_mean_pool
        elif graph_pooling == "max":  self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":  self.pool = Set2Set(emb_dim, processing_steps=2)
        else: raise ValueError("Invalid graph pooling type.")

        # transformer components
        self.tf_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim, nhead = 4, dim_feedforward = self.emb_dim,
            dropout = 0.1, batch_first = True
        )
        self.tf_encoder = nn.TransformerEncoder(
            encoder_layer=self.tf_enc_layer, num_layers=2,
        )
        assert self.emb_dim % 4 == 0

        # prediction heads
        self.pred_number_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim*2),
            nn.ReLU(),
            nn.Linear(self.emb_dim*2, 15+1),   # 默认最多9个位点
        )
        self.pred_position_layer = nn.Sequential(
            nn.Linear(self.emb_dim, int(self.emb_dim/4)), 
            nn.ReLU(),
            nn.Linear(int(self.emb_dim/4), 36),
        )
        self.pred_height_layer = nn.Sequential(
            nn.Linear(self.emb_dim, int(self.emb_dim/4)), 
            nn.ReLU(),
            nn.Linear(int(self.emb_dim/4), 1),
        )

    def get_node_feature(self, molecule_features, batch_index):
        # 将pytorch_geometric中压缩成一团的node_feature还原为 [b, max_node, dim]形式
        this_batch_size = int(batch_index[-1]) + 1
        index_list = index_transform(
            raw_index=batch_index, batch_size= this_batch_size,
            max_node_num = self.max_node_num,
        )
        new_batch_list = []
        for batch_id in range(this_batch_size):
            empty_batch_tensor = torch.zeros(self.max_node_num, self.emb_dim)
            for i in range(len(index_list[batch_id])):
                empty_batch_tensor[i, :] = molecule_features[index_list[batch_id][i], :]
            new_batch_list.append(empty_batch_tensor)
        node_feature = torch.stack(new_batch_list, dim=0).to(molecule_features.device)

        return node_feature, index_list

    def graph_encoder(self, batched_atom_bond, batched_bond_angle):
        # input: batched_atom_bond: node feature, batched_bond_angle: edge feature

        if self.args.Use_geometry_enhanced==True:
            h_node, h_node_ba= self.gnn_node(batched_atom_bond, batched_bond_angle)
        else:
            h_node = self.gnn_node(batched_atom_bond, batched_bond_angle)

        # get node_feature and total_feature(h_graph)
        node_feature, node_index = self.get_node_feature(h_node, batched_atom_bond.batch)
        h_graph = self.pool(h_node, batched_atom_bond.batch).unsqueeze(1) # [batch, 1, self.emb_dim]
        total_node_feature = torch.cat([h_graph, node_feature], dim=1)

        # get feature padding masks
        node_padding_mask = feat_padding_mask(index = node_index, max_node_num = self.max_node_num)
        pooling_padding_mask = torch.zeros(node_padding_mask.size(0), 1).to(torch.float32)
        total_padding_mask = torch.cat([pooling_padding_mask, node_padding_mask], dim=1).to(total_node_feature.device)

        return total_node_feature, total_padding_mask

    def forward(self, batched_atom_bond, batched_bond_angle, query_padding_mask=None):
        node_feat, node_padding_mask = self.graph_encoder(batched_atom_bond, batched_bond_angle)

        # get input queries
        query_single_feat = self.query_embed.weight.unsqueeze(0)
        query_feat = query_single_feat.repeat(node_feat.size(0), 1, 1)

        # get peak number prediction
        node_summary_feat = node_feat[:, 0, :]                          # [batch, emb_dim]
        pred_number_output = F.softmax(self.pred_number_layer(node_summary_feat))

        if not self.training:
            # During inference, we cannot get peak_num and query_padding_mask from groundtruth.
            # Get input query padding masks from model prediction
            pred_peak_num = pred_number_output.argmax(dim=1)
            peak_position = [[1]*pred_peak_num[i]+[-1]*(15-pred_peak_num[i]) for i in range(len(pred_peak_num))]
            peak_position = torch.tensor(peak_position)
            query_padding_mask = get_key_padding_mask(peak_position).to(node_feat.device)

        # concat the feats and padding_masks from query & node
        encoder_input_feat = torch.cat([node_feat, query_feat], dim=1)
        encoder_padding_mask = torch.cat([node_padding_mask, query_padding_mask], dim=1).to(node_feat.device)
        
        # compute from the transformer encoder, get height & position prediction
        output = self.tf_encoder(src=encoder_input_feat, src_key_padding_mask=encoder_padding_mask)
        node_feat_output = output[:, :node_feat.size(1), :]             # [batch, 64,emb_dim]
        query_output = output[:, node_feat.size(1):, :]                 # [batch, 15, emb_dim]
        pred_position_output = self.pred_position_layer(query_output)   # [batch, 15, 36]
        pred_height_output = self.pred_height_layer(query_output)   # [batch, 15, 1]

        if self.training:
            # return pred_position_output.squeeze(2), pred_number_output
            return pred_position_output.squeeze(2), pred_height_output, pred_number_output
        else: # inference time
            attn_weights = torch.einsum("bid,bjd->bij", [node_feat_output, query_output[:,0,:].unsqueeze(1)]) # [batch, 64, 1]
            attn_weights = attn_weights[:, 1:, :].squeeze()
            attn_mask = node_padding_mask[:, 1:]
            # return pred_position_output, torch.clamp(pred_number_output, min=0, max=1e8), \
            #        {"attn": attn_weights.to('cpu').tolist(), "attn_mask": attn_mask.to('cpu').tolist(),\
            #            "peak_prob": pred_number_output.to('cpu').tolist()}
            return pred_position_output, pred_height_output, torch.clamp(pred_number_output, min=0, max=1e8), \
                   {"attn": attn_weights.to('cpu').tolist(), "attn_mask": attn_mask.to('cpu').tolist(),\
                       "peak_prob": pred_number_output.to('cpu').tolist()}


def train_GNNIR(args, model, device, loader_atom_bond, loader_bond_angle, optimizer):
    model.train()
    loss_accum = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss(reduction='mean')

    for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
        batch_atom_bond, batch_bond_angle = batch[0], batch[1]
        batch_atom_bond, batch_bond_angle = batch_atom_bond.to(device), batch_bond_angle.to(device)
        # batch_atom_bond, batch_bond_angle = batch_atom_bond.cuda(), batch_bond_angle.cuda()

        # training the model for height, position, and number
        pos_pred, height_pred, num_pred = model(
            batch_atom_bond, batch_bond_angle, batch_atom_bond.query_mask,
        )
        pos_gt, height_gt = batch_atom_bond.peak_position, batch_atom_bond.peak_height
        num_gt = batch_atom_bond.peak_num

        # transform the groundtruth and prediction labels
        new_gt_pos, new_pred_pos = [], []
        new_gt_height, new_pred_height = [], []
        for i in range(batch_atom_bond.peak_num.size(0)):
            new_gt_pos.append(pos_gt[i, :int(batch_atom_bond.peak_num[i])])
            new_pred_pos.append(pos_pred[i, :int(batch_atom_bond.peak_num[i])])
            new_gt_height.append(height_gt[i, :int(batch_atom_bond.peak_num[i])])
            new_pred_height.append(height_pred[i, :int(batch_atom_bond.peak_num[i])])
        
        new_gt_pos_tensor = torch.cat(new_gt_pos, dim=0)            # [batch*node_num]
        new_pred_pos_tensor = torch.cat(new_pred_pos, dim=0)        # [batch*node_num]
        new_gt_height_tensor = torch.cat(new_gt_height, dim=0)      # [batch*node_num]
        new_pred_height_tensor=torch.cat(new_pred_height, dim=0)    # [batch*node_num]
        
        # backward propagation
        loss_pos = ce_loss(new_pred_pos_tensor, new_gt_pos_tensor)
        loss_height = mse_loss(new_pred_height_tensor, new_gt_height_tensor)
        loss_num = ce_loss(num_pred, num_gt)
        loss = loss_num + loss_pos + loss_height

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval_GNNIR(args, model, device, loader_atom_bond, loader_bond_angle):
    model.eval()
    loss_accum = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss(reduction='mean')
    total_loss_pos, total_loss_num, total_loss_height = 0, 0, 0
    
    with torch.no_grad():
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond, batch_bond_angle = batch[0], batch[1]
            batch_atom_bond, batch_bond_angle = batch_atom_bond.to(device), batch_bond_angle.to(device)
            
            pos_pred, height_pred, num_pred, _ = model(
                batch_atom_bond, batch_bond_angle, batch_atom_bond.query_mask,
            )
            pos_gt, height_gt = batch_atom_bond.peak_position, batch_atom_bond.peak_height
            num_gt = batch_atom_bond.peak_num

            # transform the groundtruth and prediction labels
            new_gt_pos, new_pred_pos = [], []
            new_gt_height, new_pred_height = [], []
            for i in range(batch_atom_bond.peak_num.size(0)):
                new_gt_pos.append(pos_gt[i, :int(batch_atom_bond.peak_num[i])])
                new_pred_pos.append(pos_pred[i, :int(batch_atom_bond.peak_num[i]), :])
                new_gt_height.append(height_gt[i, :int(batch_atom_bond.peak_num[i])])
                new_pred_height.append(height_pred[i, :int(batch_atom_bond.peak_num[i]), :])
            
            new_gt_pos_tensor = torch.cat(new_gt_pos, dim=0)            # [batch*node_num]
            new_pred_pos_tensor = torch.cat(new_pred_pos, dim=0)        # [batch*node_num, 20]
            new_gt_height_tensor = torch.cat(new_gt_height, dim=0)      # [batch*node_num]
            new_pred_height_tensor=torch.cat(new_pred_height, dim=0)    # [batch*node_num, 2]

            # calculate the loss without backward propagation
            loss_pos = ce_loss(new_pred_pos_tensor, new_gt_pos_tensor)
            loss_height = mse_loss(new_pred_height_tensor, new_gt_height_tensor)
            loss_num = ce_loss(num_pred, num_gt)
            loss = loss_num + loss_pos + loss_height

            total_loss_pos += loss_pos.detach().cpu().item()
            total_loss_num += loss_num.detach().cpu().item()
            total_loss_height += loss_height.detach().cpu().item()
            loss_accum += loss.detach().cpu().item()

    print(f"LOSS: pos={total_loss_pos}, num={total_loss_num}, height={total_loss_height}")  
    return loss_accum / (step + 1)

def test_GNNIR(args, model, device, loader_atom_bond, loader_bond_angle, epoch=-1):
    model.eval()
    total_groundtruth, total_prediction = [], []
    pos_results, atom_attn_weights_result = [], []
    num_pred_list, num_gt_list, num_rmse_list = [], [], []
    acc, acc_1, acc_2, acc_3, total_case_num = 0, 0, 0, 0, 0

    with torch.no_grad():
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond, batch_bond_angle = batch[0], batch[1]
            batch_atom_bond, batch_bond_angle = batch_atom_bond.to(device), batch_bond_angle.to(device)
            
            pos_pred, height_pred, prob_num, attn_info = model(
                batch_atom_bond, batch_bond_angle, batch_atom_bond.query_mask,
            )
            pos_gt, height_gt = batch_atom_bond.peak_position, batch_atom_bond.peak_height
            num_gt = batch_atom_bond.peak_num

            # get number groundtruth and prediction
            pred_peak_num = torch.argmax(prob_num, dim=1)
            num_pred_list.append(pred_peak_num.detach().cpu()); num_gt_list.append(num_gt.detach().cpu())
            a0, a1, a2, a3 = Accuracy(pred_peak_num, num_gt)
            acc += a0
            acc_1 += a1
            acc_2 += a2
            acc_3 += a3
            total_case_num += num_gt.size(0)

            # transform the groundtruth and prediction labels
            batch_pos_gt, batch_pos_pred = [], []
            batch_height_gt, batch_height_pred = [], []
            tmp_pos_gt, tmp_pos_pred = [], []

            for i in range(batch_atom_bond.peak_num.size(0)):
                # save the position prediction and groundtruth
                pos_gt_labels = pos_gt[i, :int(batch_atom_bond.peak_num[i])].tolist()
                pos_pred_labels = torch.argmax(pos_pred[i, :int(pred_peak_num[i]), :], dim=1).tolist()

                batch_pos_gt.append(pos_gt_labels)
                batch_pos_pred.append(pos_pred_labels)
                tmp_pos_gt.extend(pos_gt_labels[:min(len(pos_gt_labels), len(pos_pred_labels))])
                tmp_pos_pred.extend(pos_pred_labels[:min(len(pos_gt_labels), len(pos_pred_labels))])

                # save the height prediction and groundtruth
                height_gt_score = height_gt[i, :int(batch_atom_bond.peak_num[i])].tolist()
                height_pred_score = height_pred[i, :int(pred_peak_num[i]), :].squeeze().tolist()

                batch_height_gt.append(height_gt_score)
                batch_height_pred.append(height_pred_score)

            new_rmse_position = np.sqrt(mean_squared_error(tmp_pos_gt, tmp_pos_pred))
            pos_results.append(new_rmse_position)
            atom_attn_weights_result.append(attn_info)

            total_groundtruth.append({'pos': batch_pos_gt, 'height': batch_height_gt})
            total_prediction.append({'pos': batch_pos_pred, 'height': batch_height_pred})
    
    # calculate the peak number prediction accuracy
    peak_num_acc_list = dict(
        acc = acc / total_case_num, acc_1 = acc_1 / total_case_num,
        acc_2 = acc_2 / total_case_num, acc_3 = acc_3 / total_case_num,
    )

    # # show the height first match number
    # total_first_num, correct_first_num = 0, 0
    # total_num, correct_num = 0, 0
    # for batch_id in range(len(total_groundtruth)):
    #     height_gt, height_pred = total_groundtruth[batch_id]['height'], total_prediction[batch_id]['height']
    #     assert len(height_gt) == len(height_pred)
    #     for sample_id in range(len(height_gt)):
    #         if min(len(height_gt[sample_id]), len(height_pred[sample_id])) == 0: continue

    #         if height_gt[sample_id][0] == height_pred[sample_id][0]: correct_first_num += 1
    #         total_num += min(len(height_gt[sample_id]), len(height_pred[sample_id]))
    #         for i in range(min(len(height_gt[sample_id]), len(height_pred[sample_id]))):
    #             if height_gt[sample_id][i] == height_pred[sample_id][i]: correct_num += 1 
    #     total_first_num += len(height_gt)
    
    height_rmse_list = []
    for batch_id in range(len(total_groundtruth)):
        height_gt, height_pred = total_groundtruth[batch_id]['height'], total_prediction[batch_id]['height']
        assert len(height_gt) == len(height_pred)
        for case_id in range(len(height_gt)):
            min_len = min(len(height_gt[case_id]), len(height_pred[case_id]))
            height_rmse_list.append(
                np.sqrt(mean_squared_error(height_gt[case_id][:min_len], height_pred[case_id][:min_len]))
            )
    height_rmse = sum(height_rmse_list)/len(height_rmse_list)       

    for i in range(len(num_gt_list)):
        tmp_num_rmse = np.sqrt(mean_squared_error(num_gt_list[i], num_pred_list[i]))
        num_rmse_list.append(tmp_num_rmse)
    
    new_pos = sum(pos_results)/len(pos_results)
    num_rmse = sum(num_rmse_list)/len(num_rmse_list)

    print("epoch: {}, new_pos = {}, num = {}, num_rmse = {}, height_rmse = {}\n".format(
        epoch, new_pos, peak_num_acc_list, num_rmse, height_rmse,
    ))

    # total_groundtruth, total_prediction
    # total_groundtruth: 一个list, 每个item是一个batch的信息, itm={'pos':[[], [] ...], 'height':[[], [] ...]}
    return total_groundtruth, total_prediction, {'new_pos': new_pos, 'attn': atom_attn_weights_result}

def pred_real_GNNIR(args, model, device, loader_atom_bond, loader_bond_angle, epoch=-1):
    model.eval()
    batch_pos_pred, batch_height_pred = [], []

    with torch.no_grad():
        for step, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond, batch_bond_angle = batch[0], batch[1]
            batch_atom_bond, batch_bond_angle = batch_atom_bond.to(device), batch_bond_angle.to(device)
            
            pos_pred, height_pred, prob_num, attn_info = model(
                batch_atom_bond, batch_bond_angle, None,
            )
            pred_peak_num = torch.argmax(prob_num, dim=1)
            
            for i in range(pred_peak_num.size(0)):
                pos_pred_labels = torch.argmax(pos_pred[i, :int(pred_peak_num[i]), :], dim=1).tolist()
                height_pred_labels = torch.argmax(height_pred[i, :int(pred_peak_num[i]), :], dim=1).tolist()

                batch_pos_pred.append(pos_pred_labels); batch_height_pred.append(height_pred_labels)
    
    assert len(batch_pos_pred)==len(batch_height_pred)
    return batch_pos_pred, batch_height_pred