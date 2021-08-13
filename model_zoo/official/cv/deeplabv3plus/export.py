"""export AIR file."""
import argparse
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.deeplab_v3plus import DeepLabV3Plus

context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='checkpoint export')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint of deeplabv3 (Default: None)')
    parser.add_argument('--model', type=str, default='DeepLabV3plus_s8',
                        choices=['DeepLabV3plus_s16', 'DeepLabV3plus_s8'],
                        help='Select model structure (Default: DeepLabV3plus_s8)')
    parser.add_argument('--num_classes', type=int, default=21, help='the number of classes (Default: 21)')
    args = parser.parse_args()

    if args.model == 'DeepLabV3plus_s16':
        network = DeepLabV3Plus('eval', args.num_classes, 16, True)
    else:
        network = DeepLabV3Plus('eval', args.num_classes, 8, True)
    param_dict = load_checkpoint(args.checkpoint)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    input_data = np.random.uniform(0.0, 1.0, size=[32, 3, 513, 513]).astype(np.float32)
    export(network, Tensor(input_data), file_name=args.model + '-300_11.air', file_format='AIR')
