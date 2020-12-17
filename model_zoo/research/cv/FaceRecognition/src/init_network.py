# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""init network"""
import math
import numpy as np

import mindspore.nn as nn
from mindspore.common.initializer import initializer

from src.backbone.resnet import IRBlock
from src import metric_factory
from src import me_init

np.random.seed(1)

def init_net(args, network):
    '''init_net'''
    for name, cell in network.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            find_flag = True
            if cell.weight is not None:
                cell.weight.set_data(initializer(me_init.ReidKaimingUniform(a=math.sqrt(5), mode='fan_out'),
                                                 cell.weight.shape))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape))

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.weight} {cell.bias}')
            args.logger.info('---------------{}---------------'.format(find_info))

        elif isinstance(cell, nn.Dense):
            find_flag = True
            if cell.weight is not None:
                cell.weight.set_data(initializer(me_init.ReidKaimingNormal(a=math.sqrt(5), mode='fan_out'),
                                                 cell.weight.shape))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape))

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.weight} {cell.bias}')
            args.logger.info('---------------{}---------------'.format(find_info))

        elif isinstance(cell, nn.BatchNorm2d):
            find_flag = True
            # defulat gamma 1 and beta 0, and if you set should be careful for the IRBlock gamma value

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.gamma} {cell.beta}')
            args.logger.info('---------------{}---------------'.format(find_info))
        elif isinstance(cell, nn.BatchNorm1d):
            pass
        elif isinstance(cell, metric_factory.CombineMarginFC):
            find_flag = True
            if cell.weight is not None:
                cell.weight.set_data(initializer(me_init.ReidKaimingUniform(a=math.sqrt(5), mode='fan_out'),
                                                 cell.weight.shape))

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.weight}')
            args.logger.info('---------------{}---------------'.format(find_info))
        elif isinstance(cell, nn.PReLU):
            find_flag = True

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.w}')
            args.logger.info('---------------{}---------------'.format(find_info))
        elif isinstance(cell, IRBlock):
            find_flag = True
            # be careful for bn3 Do not change the name unless IRBlock last bn change name
            cell.bn3.gamma.set_data(initializer('zeros', cell.bn3.gamma.shape))

            if find_flag:
                find_info = 'PARAMETER FIND'
            else:
                find_info = 'PARAMETER UNFIND'

            args.logger.info('---------------{}---------------'.format(find_info))
            args.logger.info(f'{name} --> {cell.bn3.gamma}')
            args.logger.info('---------------{}---------------'.format(find_info))
