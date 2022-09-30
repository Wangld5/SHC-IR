import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn.modules as nn
import pdb
from models.VTS import *


class ResNet(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layer = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                           self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.tanh = nn.Tanh()
        self.label_linear = nn.Linear(label_size, hash_bit)
        self.dropout = nn.Dropout(0.5) 
        if config['without_BN']:
            self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.hash_layer.weight.data.normal_(0, 0.01)
            self.hash_layer.bias.data.fill_(0.0)
        else:
            self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.layer_hash.weight.data.normal_(0, 0.01)
            self.layer_hash.bias.data.fill_(0.0)
            self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x, T, label_vectors):

        feat = self.feature_layer(x)
        # class_feat = torch.flatten(feat, 1)
        # classification_head = self.fc(class_feat)
        feat = feat.view(feat.shape[0], -1)
        # feat = self.dropout(feat) 
        x = self.hash_layer(feat)
        x = self.tanh(x)

        return x, feat

class MoCo(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(MoCo, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet(config, hash_bit, label_size, pretrained)
        # self.encoder_k = ResNet(config, hash_bit, label_size, pretrained)
        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient
        
        # self.queue = torch.randn(config['num_train'], hash_bit)
        # self.queue = F.normalize(self.queue)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x, T, label_vectors):
        encode_x, _ = self.encoder_q(x, T, label_vectors)
        encode_x2 = None
        # with torch.no_grad():
        #     self._momentum_update_key_encoder()
        #     encode_x2, _ = self.encoder_k(x, T, label_vectors)
        return encode_x, encode_x2

class Binary_hash(nn.Module):
    def __init__(self, bits, model):
        super(Binary_hash, self).__init__()
        self.resnet50 = model
        feature_shape = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(feature_shape, bits)

    def forward(self, inputs):
        outputs = self.resnet50(inputs)
        outputs = torch.sign(outputs)
        return outputs

class ResNetClass(nn.Module):
    def __init__(self, label_size, pretrained=True):
        super(ResNetClass, self).__init__()
        self.model_resnet = models.resnet50(pretrained=pretrained)
        self.model_resnet.fc = nn.Linear(self.model_resnet.fc.in_features, label_size)
        self.BN = nn.BatchNorm1d(label_size, momentum=0.1)

    def forward(self, x):
        feat = self.model_resnet(x)
        return self.BN(feat)

class ViTClass(nn.Module):
    def __init__(self, config, label_size, img_size=224, num_cls=21843, zero_head=False, vis=True):
        super(ViTClass, self).__init__()
        self.vit = VisionTransformer(config, img_size, num_cls, zero_head, vis)
        self.fc = nn.Linear(768, label_size)
    
    def forward(self, x):
        feature, _ = self.vit(x)
        return self.fc(feature[:, 0])
    
    def load_from(self, weights):
        self.vit.load_from(weights)

class AlexNetClass(nn.Module):
    def __init__(self, label_size, pretrained=True):
        super(AlexNetClass, self).__init__()
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias
        self.cls_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096, label_size)
        self.BN = nn.BatchNorm1d(label_size, momentum=0.1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.cls_layer(x)
        feat = self.fc(x)
        return self.BN(feat)

class VGGNetClass(nn.Module):
    def __init__(self, label_size, pretrained=True):
        super(VGGNetClass, self).__init__()
        self.model_vggnet = models.vgg16(pretrained=pretrained)
        self.model_vggnet.classifier = nn.Sequential(*list(self.model_vggnet.classifier.children())[:6])
        self.fc = nn.Linear(4096, label_size)
        self.BN = nn.BatchNorm1d(label_size, momentum=0.1)

    def forward(self, x):
        x = self.model_vggnet.features(x)
        x = x.view(x.size(0), -1)
        x = self.model_vggnet.classifier(x)
        feat = self.fc(x)
        return self.BN(feat)

class weightConstrain(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(-1, 1) #将参数范围限制到-1- 1之间
            module.weight.data=w