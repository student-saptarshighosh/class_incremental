import math
import torch
from torch import nn
from torch.nn import functional as F

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        # Compute cosine similarities
        normalized_input = F.normalize(input, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        out = F.linear(normalized_input, normalized_weight)

        if self.to_reduce:
            # Reduce proxies
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768, use_init_ptm=False):
        out_all = None

        for task in range(cur_task + 1):
            if task == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (task - 1) * inc
                end_cls = start_cls + inc
            
            task_out = 0.0
            for feature_group in range(self.in_features // out_dim):
                start_feat = feature_group * out_dim
                end_feat = (feature_group + 1) * out_dim

                input_slice = F.normalize(input[:, start_feat:end_feat], p=2, dim=1)
                weight_slice = F.normalize(self.weight[start_cls:end_cls, start_feat:end_feat], p=2, dim=1)

                if use_init_ptm and feature_group == 0:
                    # PTM feature
                    group_out = beta * F.linear(input_slice, weight_slice)
                elif (use_init_ptm and feature_group != (task + 1)) or (not use_init_ptm and feature_group != task):
                    # Apply reweighting
                    group_out = (alpha / cur_task) * F.linear(input_slice, weight_slice)
                else:
                    # Normal linear operation
                    group_out = F.linear(input_slice, weight_slice)

                task_out += group_out

            if out_all is None:
                out_all = task_out
            else:
                out_all = torch.cat((out_all, task_out), dim=1)

        if self.to_reduce:
            # Reduce proxies
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all
        
        return {'logits': out_all}

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    
    batch_size = out.shape[0]
    nb_classes = out.shape[1] // nb_proxy
    
    # Reshape to group proxies for each class
    simi_per_class = out.view(batch_size, nb_classes, nb_proxy)
    
    # Compute attention weights using softmax
    attentions = F.softmax(simi_per_class, dim=-1)
    
    # Compute weighted average of proxies
    reduced = (attentions * simi_per_class).sum(-1)
    
    return reduced