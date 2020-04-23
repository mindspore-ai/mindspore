# Copyright 2019 Huawei Technologies Co., Ltd
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

"""squeeze grad"""
import _akg.topi as topi

def SqueezeGrad(y_grad, x_shape, axis=None):
    """
    Computes gradients for squeeze op.

    Args:
        y_grad (tvm.tensor.Tensor): the gradient needed to be propagation.
        x_shape (Union[list, tuple]): output Tensor shape.
        axis (Union[list, tuple, int, None], optional): eliminated axis by squeeze.

    Returns:
        tvm.tensor.Tensor: output gradient.
    """
    return topi.reshape(y_grad, x_shape)


def gpu_schedule_SqueezeGrad(outs):
    """
    gpu schedule SqueezeGrad.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    from .default_schedule import default_schedule
    return default_schedule(outs)
