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
"""train"""

import argparse
import numpy as np
import mindspore.context as context
from src.read_var import read_nc
from src.GOMO import GOMO_init, GOMO, read_init

parser = argparse.ArgumentParser(description='GOMO')
parser.add_argument('--file_path', type=str, default=None, help='file path')
parser.add_argument('--outputs_path', type=str, default=None, help='outputs path')
parser.add_argument('--im', type=int, default=65, help='im size')
parser.add_argument('--jm', type=int, default=49, help='jm size')
parser.add_argument('--kb', type=int, default=21, help='kb size')
parser.add_argument('--stencil_width', type=int, default=1, help='stencil width')
parser.add_argument('--step', type=int, default=10, help='time step')
args_gomo = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, enable_graph_kernel=True)

if __name__ == "__main__":
    variable = read_nc(args_gomo.file_path)
    im = args_gomo.im
    jm = args_gomo.jm
    kb = args_gomo.kb
    stencil_width = args_gomo.stencil_width

    # variable init
    dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, dt, h, w, wubot, wvbot, vfluxb, utb, vtb, dhb, egb, vfluxf, z, zz, \
    dzz, cor, fsm = read_init(
        variable, im, jm, kb)

    # define grid and init variable update
    net_init = GOMO_init(im, jm, kb, stencil_width)
    init_res = net_init(dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, h, w, vfluxf, zz, fsm)
    for res_tensor in init_res:
        if isinstance(res_tensor, (list, tuple)):
            for rt in res_tensor:
                rt.data_sync(True)
        else:
            res_tensor.data_sync(True)
    ua, va, el, et, etf, d, dt, l, q2b, q2lb, kh, km, kq, aam, w, q2, q2l, t, s, u, v, cbc, rmean, rho, x_d, y_d, z_d\
        = init_res

    # define GOMO model
    Model = GOMO(im=im, jm=jm, kb=kb, stencil_width=stencil_width, variable=variable, x_d=x_d, y_d=y_d, z_d=z_d,
                 q2b=q2b, q2lb=q2lb, aam=aam, cbc=cbc, rmean=rmean)

    # time step of GOMO Model
    for step in range(1, args_gomo.step+1):
        elf, etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s, rho, wubot, wvbot, ub, vb, \
        egb, etb, dt, dhb, utb, vtb, vfluxb, et, steps, vamax, q2b, q2lb = Model(
            etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s, rho,
            wubot, wvbot, ub, vb, egb, etb, dt, dhb, utb, vtb, vfluxb, et)
        vars_list = etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s, rho, wubot, wvbot, \
               ub, vb, egb, etb, dt, dhb, utb, vtb, vfluxb, et
        for var in vars_list:
            var.asnumpy()
        # save output
        if step % 5 == 0:
            np.save(args_gomo.outputs_path + "u_"+str(step)+".npy", u.asnumpy())
            np.save(args_gomo.outputs_path + "v_" + str(step) + ".npy", v.asnumpy())
            np.save(args_gomo.outputs_path + "t_" + str(step) + ".npy", t.asnumpy())
            np.save(args_gomo.outputs_path + "et_" + str(step) + ".npy", et.asnumpy())
