from .resnet import resnet
from .fcresnet import fcresnet

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'fcresnet':
        return fcresnet(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
