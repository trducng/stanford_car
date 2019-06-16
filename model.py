"""Construct the model"""

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torchvision import models



def get_resnet_feature_extractor(version=152):
    """Get the feature extractor from ResNet model

    # Arguments
        version [int]: the resnet version, should be one of the
            following [18, 34, 50, 101, 152]

    # Returns
        [nn.Module]: a sequential features
    """
    version = int(version)
    supported_versions = [18, 34, 50, 101, 152]
    if int(version) not in supported_versions:
        raise AttributeError('version should be {} but {}'.format(
            supported_versions, version))

    if version == 18:
        model = models.resnet18(pretrained=True)
    elif version == 34:
        model = models.resnet34(pretrained=True)
    elif version == 50:
        model = models.resnet50(pretrained=True)
    elif version == 101:
        model = models.resnet101(pretrained=True)
    else:
        model = models.resnet152(pretrained=True)

    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,

        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )

    return features


class BilinearAttentionPool(nn.Module):
    """Bilinear Attention Pooling.
    Source: https://arxiv.org/abs/1901.09891

    This module uses attention map to transform feature map. Specifically,
    the feature map is element-wise multiplied by an attention map. The result
    is then concatenated and reduced using convolutional or pooling operations.

    # Arguments
        pool [str]: whether 'avg' or 'max' or None
        conv [dict]: if not None, should be a dictionary containing conv
            initialization parameters
    """

    def __init__(self, pool=None, conv=None):
        """Initialize the layer"""
        super(BilinearAttentionPool, self).__init__()

        post_process = None
        if pool.lower() == 'avg':
            post_process = nn.AdaptiveAvgPool2d(1)
        elif pool.lower() == 'max':
            post_process = nn.AdaptiveMaxPool2d(1)
        elif isinstance(conv, dict):
            post_process = nn.Conv2d(**conv)
        else:
            post_process = nn.AdativeAvgPool2d(1)

        self.post_process = post_process

    def forward(self, feature_maps, attention_maps):
        """Perform the forward pass

        # Arguments
            feature_maps [4D Tensor]: shape B x C1 x H x W
            attention_maps [4D Tensor]: shape B x C2 x H x W

        # Returns
            [4D tensor]: shape B x (C1C2) x 1 x 1 
        """
        part_feature_maps = []
        n_maps = attention_maps.size(1)

        for map_idx in range(n_maps):
            part_feature_maps.append(
                feature_maps * attention_maps[:,map_idx,:,:].unsqueeze(1))

        part_feature_maps = torch.cat(part_feature_maps, dim=1)
        return self.post_process(part_feature_maps)


class Model(nn.Module):
    """The model that makes use of Weakly-Supervised Data Augmentation Network

    # Arguments
        n_classes [int]: the number of output classes
        input_size [tuple of int]: the shape of image (width x height)
        n_attentions [int]: the number of attention maps
        resnet_version [int]: support 18, 34, 50, 101 and 152
        ckpt_path [str]: the path to checkpoint model. If this value is set,
            all earlier values will be ignored
        gpu [bool]: whether to use gpu
    """

    def __init__(self, n_classes=196, input_size=(256, 256), n_attentions=4,
                 resnet_version=152, ckpt=None, gpu=False):
        """Initialize the object"""
        super(Model, self).__init__()

        self.gpu = gpu
        if ckpt is not None:
            map_location = 'cuda:0' if gpu else 'cpu'
            model_info = torch.load(ckpt, map_location=map_location)
            n_classes = model_info['n_classes']
            input_size = model_info['input_size']
            resnet_version = model_info['resnet_version']
            n_attentions = model_info['n_attentions']
            
        self.n_classes = n_classes
        self.input_size = input_size
        self.n_attentions = n_attentions

        # model information
        self.conv = get_resnet_feature_extractor(version=resnet_version)
        self.attention = nn.Conv2d(
            in_channels=2048, out_channels=n_attentions, kernel_size=1,
            padding=0, bias=False
        )
        self.bap = BilinearAttentionPool(pool='avg')
        self.output = nn.Linear(in_features=n_attentions * 2048,
                                out_features=n_classes)
        
        # load the state dict
        if ckpt is not None:
            self.load_state_dict(model_info['state_dict'])
        
    def forward(self, input_tensor):
        """Perform the forward pass
        
        # Arguments
            input_tensor [4D tensor]: shave (B x C x H x W)

        # Returns
            [2D tensor]: logit of shape (B x n_classes)
            [4D tensor]: feature map of shape (B x 2048 x H_ x W_)
            [4D tensor]: the attention map of shape (B x H_ x W_)
        """
        mini_batch = input_tensor.size(0)

        # go through feature extractor
        feature_maps = self.conv(input_tensor)
        attention_maps = self.attention(feature_maps)
        feature_matrix = self.bap(feature_maps, attention_maps)
        feature_matrix = feature_matrix.view(mini_batch, -1).contiguous()
        # feature_matrix has size (B x (2048M))
        logit = self.output(feature_matrix)

        # get a random attention map for each instance in the batch size
        random_indices = torch.randint(self.n_attentions, (mini_batch,))
        sampled_attentions = attention_maps[
            torch.arange(mini_batch),
            random_indices
        ]
        
        # normalize attention map
        # NOTE: can normalized based on knowledge of other unsampled attentions
        height, width = sampled_attentions.shape[1:]
        sampled_attentions = sampled_attentions.view(mini_batch, -1)
        min_value, _ = sampled_attentions.min(dim=1, keepdim=True)
        max_value, _ = sampled_attentions.max(dim=1, keepdim=True)
        sampled_attentions = ((sampled_attentions - min_value)
            / (max_value - min_value)).view(mini_batch, height, width)

        return logit, feature_matrix, sampled_attentions


if __name__ == '__main__':
    checkpoint = '/home/john/temp/ckpt/006_resnet101_normed'
    model = Model(ckpt=checkpoint, gpu=False)
    import pdb; pdb.set_trace()

