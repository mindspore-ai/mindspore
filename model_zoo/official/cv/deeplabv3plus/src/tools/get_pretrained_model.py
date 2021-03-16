import torch
from mindspore import Tensor, save_checkpoint


def torch2ms(pth_path, ckpt_path):
    pretrained_dict = torch.load(pth_path)
    print('--------------------pretrained keys------------------------')
    for k in pretrained_dict:
        print(k)

    print('---------------------torch2ms keys-----------------------')
    new_params = []
    for k, v in pretrained_dict.items():
        if 'fc' in k:
            continue
        if 'bn' in k or 'downsample.1' in k:
            k = k.replace('running_mean', 'moving_mean')
            k = k.replace('running_var', 'moving_variance')
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
        k = 'network.resnet.' + k
        print(k)
        param_dict = {'name': k, 'data': Tensor(v.detach().numpy())}
        new_params.append(param_dict)
    save_checkpoint(new_params, ckpt_path)


if __name__ == '__main__':
    pth = "./resnet101-5d3b4d8f.pth"
    ckpt = "./resnet.ckpt"
    torch2ms(pth, ckpt)
