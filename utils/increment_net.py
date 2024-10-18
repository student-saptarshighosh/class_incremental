import copy
import logging
import torch
from torch import nn
from basic.linear import CosineLinear
import timm
from basic import vit_ease
from easydict import EasyDict

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()

    # Pretrained Vision Transformer models from timm
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    # Handle _ease models (AdaptFormer)
    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease":
            config = EasyDict(
                # AdaptFormer configuration for ffn (feedforward network) adaptation
                ffn_adapt=True,
                ffn_option="parallel",  # Adapter applied in parallel
                ffn_adapter_layernorm_option="none",  # No layer normalization
                ffn_adapter_init_option="lora",  # Low-rank adaptation initialization
                ffn_adapter_scalar="0.1",  # Scaling factor for the adapter output
                ffn_num=ffn_num,  # Number of units in the adapter
                d_model=768,  # Input/output model dimension
                # VPT (Visual Prompt Tuning) related
                vpt_on=False,  # VPT disabled in this configuration
                vpt_num=0,
                _device=args["device"][0]
            )

            # Check the backbone model type and load the corresponding Vision Transformer with EASE adapters
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=config)
                model.out_dim = 768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=config)
                model.out_dim = 768
            else:
                raise NotImplementedError(f"Unknown type {name}")
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # Raise error for unrecognized backbone types
    else:
        raise NotImplementedError(f"Unknown type {name}")



class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('Initializing BaseNet...')
        self.backbone = get_backbone(args, pretrained)
        print('BaseNet initialized.')
        self.fc = None
        self._device = args["device"][0]

        # Determine if the backbone is a CNN or ViT-based model
        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        """
        Extract feature vectors from the backbone. 
        CNN: Extract from the 'features' field in the backbone's output dictionary.
        ViT: Directly return the backbone's output.
        """
        if self.model_type == 'cnn':
            return self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        """
        Forward pass through the network. For CNN, it uses the 'features' key,
        while for ViT, it directly uses the output.
        """
        if self.model_type == 'cnn':
            x = self.backbone(x)
            features = x['features']
            logits = self.fc(features) if self.fc else None
            out = {
                'fmaps': x.get('fmaps', []),  # Feature maps, if available
                'features': features,         # Backbone features
                'logits': logits              # Classification logits
            }
        else:
            features = self.backbone(x)
            logits = self.fc(features) if self.fc else None
            out = {
                'features': features,         # Backbone features
                'logits': logits              # Classification logits
            }

        return out

    def update_fc(self, nb_classes):
        """
        Update the fully connected layer to match the number of classes.
        If the FC layer doesn't exist, create it using the `generate_fc` method.
        """
        in_dim = self.feature_dim
        self.fc = self.generate_fc(in_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        """
        Generate a fully connected layer with specified input and output dimensions.
        """
        return nn.Linear(in_dim, out_dim).to(self._device)

    def copy(self):
        """
        Return a deep copy of the model.
        """
        return copy.deepcopy(self)

    def freeze(self):
        """
        Freeze all parameters of the model to prevent gradient updates.
        Set the model to evaluation mode.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim  # Output dimension of the backbone
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]  # Whether to use pre-trained transformer model (PTM)
        self.alpha = args["alpha"]  # Alpha value for weighted combination in reweighting
        self.beta = args["beta"]  # Beta value for PTM weighting in reweighting

    # Freezing the model parameters
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False  # Disabling gradient calculations

    # Calculate feature dimension based on the task and PTM usage
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)  # Use PTM, feature dim is multiplied accordingly
        else:
            return self.out_dim * (self._cur_task + 1)

    # Update the fully connected layer for new classes
    def update_fc(self, nb_classes):
        self._cur_task += 1  # Move to the next task
        
        # Initialize proxy_fc based on the task (init_cls for the first task, inc for subsequent tasks)
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        # Create a new fully connected layer with updated number of classes
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()  # Reset the parameters to zero for the new FC layer
        
        # If an old FC exists, transfer its parameters to the new one
        if self.fc is not None:
            old_nb_classes = self.fc.out_features  # Number of classes in the old FC layer
            weight = copy.deepcopy(self.fc.weight.data)  # Copy the old weights
            fc.sigma.data = self.fc.sigma.data  # Retain the sigma value from the old FC
            fc.weight.data[:old_nb_classes, :-self.out_dim] = nn.Parameter(weight)  # Copy old weights
        del self.fc  # Remove old FC
        self.fc = fc  # Assign new FC

    # Generate the fully connected layer
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)  # Use CosineLinear for FC layers
        return fc

    # Extract feature vectors using the backbone
    def extract_vector(self, x):
        return self.backbone(x)  # Pass through the backbone for feature extraction

    # Forward pass
    def forward(self, x, test=False):
        if not test:
            # Training mode: Extract features using the backbone and apply proxy_fc
            x = self.backbone.forward(x, False)  # Forward pass without PTM
            out = self.proxy_fc(x)  # Get logits using proxy_fc
        else:
            # Testing mode: Extract features and apply reweighting or standard FC
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)  # Forward pass with PTM
            if self.args["moni_adam"] or not self.args["use_reweight"]:
                out = self.fc(x)  # Standard FC forward pass
            else:
                # Reweighting the output logits
                out = self.fc.forward_reweight(
                    x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta
                )

        # Append features to the output dictionary
        out.update({"features": x})
        return out

    # Show the trainable parameters in the model
    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:  # Only print the trainable parameters
                print(name, param.numel())  # Output the parameter name and its number of elements


