"""hub config."""
from src.deeplab_v3plus import DeepLabV3Plus


def create_network(name, *args, **kwargs):
    freeze_bn = True
    num_classes = kwargs["num_classes"]
    if name == 'DeepLabV3plus_s16':
        DeepLabV3plus_s16_network = DeepLabV3Plus('eval', num_classes, 16, freeze_bn)
        return DeepLabV3plus_s16_network
    if name == 'DeepLabV3plus_s8':
        DeepLabV3plus_s8_network = DeepLabV3Plus('eval', num_classes, 8, freeze_bn)
        return DeepLabV3plus_s8_network
    raise NotImplementedError(f"{name} is not implemented in the repo")
