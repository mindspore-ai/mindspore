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
"""GOMO Model"""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P, functional as F
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from src.stencil import AXB, AXF, AYB, AYF, AZB, AZF
from src.stencil import DXB, DXF, DYB, DYF, DZB, DZF
from src.Grid import Grid


def read_init(variable, im, jm, kb):
    """
    read init variable from nc file

    Args:
        variable(dict): The initial variables from inputs file.
        im (int): The size of x direction.
        jm (int): The size of y direction.
        kb (int): The size of z direction.

    Returns:
        tuple[Tensor], The initial variables.
    """

    dx = Tensor(variable["dx"])
    dy = Tensor(variable["dy"])
    dz = Tensor(variable["dz"])
    tb = Tensor(variable["tb"])
    sb = Tensor(variable["sb"])
    ub = Tensor(variable["ub"])
    vb = Tensor(variable["vb"])
    uab = Tensor(variable["uab"])
    vab = Tensor(variable["vab"])
    elb = Tensor(variable["elb"])
    etb = Tensor(variable["etb"])
    dt = Tensor(variable["dt"])
    h = Tensor(variable["h"])
    vfluxf = Tensor(variable["vfluxf"])
    z = Tensor(variable["z"])
    zz = Tensor(variable["zz"])
    dzz = Tensor(variable["dzz"])
    cor = Tensor(variable["cor"])
    fsm = Tensor(variable["fsm"])
    w = Tensor(np.zeros([im, jm, kb], np.float32))
    wubot = Tensor(np.zeros([im, jm, 1], np.float32))
    wvbot = Tensor(np.zeros([im, jm, 1], np.float32))
    vfluxb = Tensor(np.zeros([im, jm, 1], np.float32))
    utb = Tensor(np.zeros([im, jm, 1], np.float32))
    vtb = Tensor(np.zeros([im, jm, 1], np.float32))
    dhb = Tensor(np.zeros([im, jm, 1], np.float32))
    egb = Tensor(np.zeros([im, jm, 1], np.float32))

    return dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, dt, h, w, wubot, wvbot, vfluxb, utb, vtb, dhb, \
           egb, vfluxf, z, zz, dzz, cor, fsm


class Shift(nn.Cell):
    """
    Shift operations
    """
    def __init__(self, dims):
        super(Shift, self).__init__()

        pad_list = ()
        self.slice_list = ()
        for i, _ in enumerate(dims):
            if dims[i] >= 0:
                pad_list += ((dims[i], 0),)
                self.slice_list += (0,)
            else:
                pad_list += ((0, -dims[i]),)
                self.slice_list += (-dims[i],)
        self.pad = P.Pad((pad_list))
        self.slice = P.Slice()
        self.shape = P.Shape()

    def construct(self, x):
        """construct"""
        x1 = self.pad(x)
        x_shape = self.shape(x)
        x1 = self.slice(x1, self.slice_list, x_shape)
        return x1


class GOMO_init(nn.Cell):
    """
    Get Ocean Model GOMO init variables
    """

    def __init__(self, im, jm, kb, stencil_width):
        super(GOMO_init, self).__init__()
        self.Grid = Grid(im, jm, kb)
        self.im = im
        self.jm = jm
        self.kb = kb
        self.kbm1 = self.kb - 1
        self.kbm2 = self.kb - 2
        self.imm1 = self.im - 1
        self.jmm1 = self.jm - 1
        self.grav = 9.8060
        self.rhoref = 1025.0
        self.z0b = 0.01
        self.small = 1e-9
        self.tbias = 0.0
        self.sbias = 0.0
        self.aam_init = 500.0
        self.cbcmax = 1.0
        self.cbcmin = 0.0025
        self.kappa = 0.40
        self.mat_ones = Tensor(np.float32(np.ones([self.im, self.jm, self.kb])))
        self.rmean = Parameter(Tensor(np.zeros([self.im, self.jm, self.kb], dtype=np.float32)), name="rmean",
                               requires_grad=False)

        self.AXB = AXB(stencil_width=stencil_width)
        self.AXF = AXF(stencil_width=stencil_width)
        self.AYB = AYB(stencil_width=stencil_width)
        self.AYF = AYF(stencil_width=stencil_width)
        self.AZB = AZB(stencil_width=stencil_width)
        self.AZF = AZF(stencil_width=stencil_width)
        self.DXB = DXB(stencil_width=stencil_width)
        self.DXF = DXF(stencil_width=stencil_width)
        self.DYB = DYB(stencil_width=stencil_width)
        self.DYF = DYF(stencil_width=stencil_width)
        self.DZB = DZB(stencil_width=stencil_width)
        self.DZF = DZF(stencil_width=stencil_width)
        self.abs = P.Abs()
        self.sqrt = P.Sqrt()
        self.log = P.Log()
        self.csum = P.CumSum()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.assign = P.Assign()
        self.pow = P.Pow()

        shape3 = (self.im, self.jm, self.kb)
        z_h = np.zeros(shape3, dtype=np.float32)
        z_h[:, :, 0] = 1.
        self.z_h = Tensor(z_h)
        z_e = np.zeros(shape3, dtype=np.float32)
        z_e[:, :, -1] = 1.
        self.z_e = Tensor(z_e)
        z_he1 = np.zeros(shape3, dtype=np.float32)
        z_he1[:, :, 0:self.kb - 1] = 1.
        self.z_he1 = Tensor(z_he1)

    def dens(self, si, ti, zz, h, fsm):
        """
        density compute function
        Args:
            si: Salinity.
            ti: Potential temperature.
            zz: sigma coordinate, intermediate between Z.
            h: the bottom depth (m)
            fsm: Mask for scalar variables; = 0 over land; = 1 over water
        Returns:
            rhoo[Tensor], density.
        """
        tr = ti + self.tbias
        sr = si + self.sbias
        tr2 = tr * tr
        tr3 = tr2 * tr
        tr4 = tr3 * tr
        p = self.grav * self.rhoref * (-1 * zz * self.mat_ones * h) * 1e-5
        rhor2 = -0.157406e0 + 6.793952e-2 * tr - 9.095290e-3 * tr2 + 1.001685e-4 * tr3 - 1.120083e-6 * tr4 + \
                6.536332e-9 * tr4 * tr
        rhor1 = rhor2 + (0.824493e0 - 4.0899e-3 * tr + 7.6438e-5 * tr2 - 8.2467e-7 * tr3 + 5.3875e-9 * tr4) * sr + \
                (-5.72466e-3 + 1.0227e-4 * tr - 1.6546e-6 * tr2) * self.abs(sr) ** 1.5e0 + 4.8314e-4 * sr * sr
        cr1 = 1449.1e0 + .0821e0 * p + 4.55e0 * tr - .045e0 * tr2 + 1.34e0 * (sr - 35.e0)
        cr = p / (cr1 * cr1)
        rhor = rhor1 + 1.e5 * cr * (1.e0 - 2.e0 * cr)
        rhoo = rhor / self.rhoref * fsm
        return rhoo

    def bottom_friction(self, zz1, h):
        """
        bottom_friction
        """
        zz_kbm1 = P.Slice()(zz1, (self.kb - 2,), (1,))
        cbc = (self.kappa / self.log((1.0 + zz_kbm1) * h / self.z0b)) * (self.kappa / self.log((1.0 + zz_kbm1) * h
                                                                                               / self.z0b))
        cbc_compare = P.Cast()(cbc > self.cbcmax, mstype.float32)
        cbc = cbc * (1 - cbc_compare) + cbc_compare * self.cbcmax
        cbc_compare = P.Cast()(cbc < self.cbcmin, mstype.float32)
        cbc = cbc * (1 - cbc_compare) + cbc_compare * self.cbcmin
        return cbc

    def construct(self, dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, h, w, vfluxf, zz, fsm):
        """construct"""
        x_d, y_d, z_d = self.Grid(dx, dy, dz)

        rho = self.dens(sb, tb, zz, h, fsm)
        rmean = self.rmean * (1 - self.z_he1) + rho * self.z_he1
        zz1 = zz[0, 0, :]
        ua = uab
        va = vab
        el = elb
        et = etb
        etf = et
        d = h + el
        dt = h + et
        l = dt * 0.1e0
        q2b = self.mat_ones * self.small
        q2lb = l * q2b
        kh = l * self.sqrt(q2b)
        km = kh
        kq = kh
        aam = self.mat_ones * self.aam_init
        w = w * (1 - self.z_h) + self.z_h * vfluxf
        q2 = q2b
        q2l = q2lb
        t = tb
        s = sb
        u = ub
        v = vb
        cbc = self.bottom_friction(zz1, h)

        return ua, va, el, et, etf, d, dt, l, q2b, q2lb, kh, km, kq, aam, w, q2, q2l, t, s, u, v, cbc, rmean, rho, \
               x_d, y_d, z_d


class GOMO(nn.Cell):
    """
    Get Ocean Model GOMO
    """

    def __init__(self, im, jm, kb, stencil_width, variable, x_d, y_d, z_d, q2b, q2lb, aam, cbc, rmean):
        super(GOMO, self).__init__()
        self.x_d = x_d
        self.y_d = y_d
        self.z_d = z_d
        self.q2b_init = q2b
        self.q2lb_init = q2lb
        self.aam_init = aam
        self.cbc_init = cbc
        self.rmean_init = rmean
        self.im = im
        self.jm = jm
        self.kb = kb
        self.kbm1 = self.kb - 1
        self.kbm2 = self.kb - 2
        self.imm1 = self.im - 1
        self.jmm1 = self.jm - 1
        self.fclim_flag = False
        self._parameter_init(variable)
        self._boundary_process()
        self._constant_init()

        self.AXB = AXB(stencil_width=stencil_width)
        self.AXF = AXF(stencil_width=stencil_width)
        self.AYB = AYB(stencil_width=stencil_width)
        self.AYF = AYF(stencil_width=stencil_width)
        self.AZB = AZB(stencil_width=stencil_width)
        self.AZF = AZF(stencil_width=stencil_width)
        self.DXB = DXB(stencil_width=stencil_width)
        self.DXF = DXF(stencil_width=stencil_width)
        self.DYB = DYB(stencil_width=stencil_width)
        self.DYF = DYF(stencil_width=stencil_width)
        self.DZB = DZB(stencil_width=stencil_width)
        self.DZF = DZF(stencil_width=stencil_width)
        self.abs = P.Abs()
        self.sqrt = P.Sqrt()
        self.log = P.Log()
        self.csum = P.CumSum()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.assign = P.Assign()
        self.pow = P.Pow()

    def _boundary_process(self,):
        """boundary process"""
        shape2 = (self.im, self.jm, 1)
        shape3 = (self.im, self.jm, self.kb)

        # 2d boundary
        x_he = np.zeros(shape2, dtype=np.float32)
        x_he[[0, -1], :, :] = 1
        self.x_he = Tensor(x_he)
        x_h = np.zeros(shape2, dtype=np.float32)
        x_h[0, :, :] = 1.
        self.x_h = Tensor(x_h)
        x_e = np.zeros(shape2, dtype=np.float32)
        x_e[-1, :, :] = 1.
        self.x_e = Tensor(x_e)
        x_h1 = np.zeros(shape2, dtype=np.float32)
        x_h1[1, :, :] = 1.
        self.x_h1 = Tensor(x_h1)
        x_e1 = np.zeros(shape2, dtype=np.float32)
        x_e1[-2, :, :] = 1.
        self.x_e1 = Tensor(x_e1)
        y_he = np.zeros(shape2, dtype=np.float32)
        y_he[:, [0, -1], :] = 1.
        self.y_he = Tensor(y_he)
        y_h = np.zeros(shape2, dtype=np.float32)
        y_h[:, 0, :] = 1.
        self.y_h = Tensor(y_h)
        y_e = np.zeros(shape2, dtype=np.float32)
        y_e[:, -1, :] = 1.
        self.y_e = Tensor(y_e)
        y_h1 = np.zeros(shape2, dtype=np.float32)
        y_h1[:, 1, :] = 1.
        self.y_h1 = Tensor(y_h1)
        y_e1 = np.zeros(shape2, dtype=np.float32)
        y_e1[:, -2, :] = 1.
        self.y_e1 = Tensor(y_e1)

        # 3d boundary
        x_h3d = np.zeros(shape3, dtype=np.float32)
        x_h3d[0, :, :] = 1.
        self.x_h3d = Tensor(x_h3d)
        x_e3d = np.zeros(shape3, dtype=np.float32)
        x_e3d[-1, :, :] = 1.
        self.x_e3d = Tensor(x_e3d)
        x_he3d = np.zeros(shape3, dtype=np.float32)
        x_he3d[[0, -1], :, :] = 1
        self.x_he3d = Tensor(x_he3d)
        x_e13d = np.zeros(shape3, dtype=np.float32)
        x_e13d[-2, :, :] = 1.
        self.x_e13d = Tensor(x_e13d)
        x_h13d = np.zeros(shape3, dtype=np.float32)
        x_h13d[1, :, :] = 1.
        self.x_h13d = Tensor(x_h13d)
        y_h3d = np.zeros(shape3, dtype=np.float32)
        y_h3d[:, 0, :] = 1.
        self.y_h3d = Tensor(y_h3d)
        y_e3d = np.zeros(shape3, dtype=np.float32)
        y_e3d[:, -1, :] = 1.
        self.y_e3d = Tensor(y_e3d)
        y_e13d = np.zeros(shape3, dtype=np.float32)
        y_e13d[:, -2, :] = 1.
        self.y_e13d = Tensor(y_e13d)
        y_he3d = np.zeros(shape3, dtype=np.float32)
        y_he3d[:, [0, -1], :] = 1.
        self.y_he3d = Tensor(y_he3d)
        y_h13d = np.zeros(shape3, dtype=np.float32)
        y_h13d[:, 1, :] = 1.
        self.y_h13d = Tensor(y_h13d)
        z_h = np.zeros(shape3, dtype=np.float32)
        z_h[:, :, 0] = 1.
        self.z_h = Tensor(z_h)
        z_h1 = np.zeros(shape3, dtype=np.float32)
        z_h1[:, :, 1] = 1.
        self.z_h1 = Tensor(z_h1)
        z_e = np.zeros(shape3, dtype=np.float32)
        z_e[:, :, -1] = 1.
        self.z_e = Tensor(z_e)
        z_e1 = np.zeros(shape3, dtype=np.float32)
        z_e1[:, :, -2] = 1.
        self.z_e1 = Tensor(z_e1)
        z_he = np.zeros(shape3, dtype=np.float32)
        z_he[:, :, [0, -1]] = 1.
        self.z_he = Tensor(z_he)
        z_he1 = np.zeros(shape3, dtype=np.float32)
        z_he1[:, :, 0:self.kb - 1] = 1.
        self.z_he1 = Tensor(z_he1)
        z_he2 = np.zeros(shape3, dtype=np.float32)
        z_he2[:, :, 0:self.kb - 2] = 1.
        self.z_he2 = Tensor(z_he2)
        z_he3 = np.zeros(shape3, dtype=np.float32)
        z_he3[:, :, 1:self.kb - 2] = 1.
        self.z_he3 = Tensor(z_he3)

        self.zslice = ()
        for i in range(self.kb):
            z_k = np.zeros(shape3, dtype=np.float32)
            z_k[:, :, i] = 1.0
            self.zslice += (Tensor(z_k),)

    def _constant_init(self,):
        """constant init"""
        self.small = 1e-9
        self.tbias = 0.0
        self.sbias = 0.0
        self.grav = 9.8059999999999992
        self.kappa = 0.40
        self.rhoref = 1025.0
        self.dte = 6.0
        self.isplit = 30
        self.dti = self.dte * self.isplit
        self.dte2 = self.dte * 2e0
        self.dti2 = self.dti * 2e0
        self.z0b = 0.01
        self.npg = 1
        self.nadv = 1
        self.ramp = 1.0
        self.horcon = 0.20
        self.umol = 2.e-5
        self.aam_init = 500.0
        self.nbct = 1
        self.nbcs = 1
        self.hmax = 4500.0
        self.nsbdy = 1
        self.ispi = 1.0 / self.isplit
        self.isp2i = self.ispi / 2.0
        self.ispadv = 5
        self.alpha = 0.2250
        self.smoth = 0.100
        self.vmaxl = 100.0
        self.cbcmax = 1.0
        self.cbcmin = 0.0025
        self.tprni = 0.20
        self.small_debug = 1e-15
        self.matrix_small = Tensor(self.small * np.ones([self.im, self.jm, self.kb]), mstype.float32)
        self.matrix_grav = Tensor(self.grav * np.ones([self.im, self.jm, self.kb]), mstype.float32)
        self.matrix2d_grav = Tensor(self.grav * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix1_dti2 = Tensor(self.dti2 * np.ones([self.im, self.jm, self.kb]), mstype.float32)
        self.matrix2_dti2 = Tensor(2.0 * self.dti2 * np.ones([self.im, self.jm, self.kb]), mstype.float32)
        self.matrix_ispi = Tensor(self.ispi * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix_isp2i = Tensor(self.isp2i * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix2_dte = Tensor(2.0 * self.dte * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix_ramp = Tensor(self.ramp * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix_smoth = Tensor(0.5 * self.smoth * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.matrix_alpha = Tensor(self.alpha * np.ones([self.im, self.jm, 1]), mstype.float32)
        self.mat_ones = Tensor(np.float32(np.ones([self.im, self.jm, self.kb])))
        self.mat_ones_2d = Tensor(np.float32(np.ones([self.im, self.jm, 1])))
        self.mat_twos_2d = Tensor(2 * np.float32(np.ones([self.im, self.jm, 1])))
        self.mat_zeros = Tensor(np.float32(np.zeros([self.im, self.jm, self.kb])))
        self.mat_zeros_im_jm_1 = Tensor(np.float32(np.zeros([self.im, self.jm, 1])))

    def _parameter_init(self, variable):
        """parameter init"""
        self.z = Parameter(Tensor(variable["z"]), name="z", requires_grad=False)
        self.zz = Parameter(Tensor(variable["zz"]), name="zz", requires_grad=False)
        self.dzz = Parameter(Tensor(variable["dzz"]), name="dzz", requires_grad=False)
        self.dx = Parameter(Tensor(variable["dx"]), name="dx", requires_grad=False)
        self.dy = Parameter(Tensor(variable["dy"]), name="dy", requires_grad=False)
        self.dz = Parameter(Tensor(variable["dz"]), name="dz", requires_grad=False)
        self.cor = Parameter(Tensor(variable["cor"]), name="cor", requires_grad=False)
        self.h = Parameter(Tensor(variable["h"]), name="h", requires_grad=False)
        self.fsm = Parameter(Tensor(variable["fsm"]), name="fsm", requires_grad=False)
        self.dum = Parameter(Tensor(variable["dum"]), name="dum", requires_grad=False)
        self.dvm = Parameter(Tensor(variable["dvm"]), name="dvm", requires_grad=False)
        self.art = Parameter(Tensor(variable["art"]), name="art", requires_grad=False)
        self.aru = Parameter(Tensor(variable["aru"]), name="aru", requires_grad=False)
        self.arv = Parameter(Tensor(variable["arv"]), name="arv", requires_grad=False)
        self.rfe = Parameter(Tensor(variable["rfe"]), name="rfe", requires_grad=False)
        self.rfw = Parameter(Tensor(variable["rfw"]), name="rfw", requires_grad=False)
        self.rfn = Parameter(Tensor(variable["rfn"]), name="rfn", requires_grad=False)
        self.rfs = Parameter(Tensor(variable["rfs"]), name="rfs", requires_grad=False)
        self.east_e = Parameter(Tensor(variable["east_e"]), name="east_e", requires_grad=False)
        self.north_e = Parameter(Tensor(variable["north_e"]), name="north_e", requires_grad=False)
        self.east_c = Parameter(Tensor(variable["east_c"]), name="east_c", requires_grad=False)
        self.north_c = Parameter(Tensor(variable["north_c"]), name="north_c", requires_grad=False)
        self.east_u = Parameter(Tensor(variable["east_u"]), name="east_u", requires_grad=False)
        self.north_u = Parameter(Tensor(variable["north_u"]), name="north_u", requires_grad=False)
        self.east_v = Parameter(Tensor(variable["east_v"]), name="east_v", requires_grad=False)
        self.north_v = Parameter(Tensor(variable["north_v"]), name="north_v", requires_grad=False)
        self.tclim = Parameter(Tensor(variable["tclim"]), name="tclim", requires_grad=False)
        self.sclim = Parameter(Tensor(variable["sclim"]), name="sclim", requires_grad=False)
        self.rot = Parameter(Tensor(variable["rot"]), name="rot", requires_grad=False)
        self.vfluxf = Parameter(Tensor(variable["vfluxf"]), name="vfluxf", requires_grad=False)
        self.wusurf = Parameter(Tensor(variable["wusurf"]), name="wusurf", requires_grad=False)
        self.wvsurf = Parameter(Tensor(variable["wvsurf"]), name="wvsurf", requires_grad=False)
        self.e_atmos = Parameter(Tensor(variable["e_atmos"]), name="e_atmos", requires_grad=False)
        self.uabw = Parameter(Tensor(variable["uabw"]), name="uabw", requires_grad=False)
        self.uabe = Parameter(Tensor(variable["uabe"]), name="uabe", requires_grad=False)
        self.vabs = Parameter(Tensor(variable["vabs"]), name="vabs", requires_grad=False)
        self.vabn = Parameter(Tensor(variable["vabn"]), name="vabn", requires_grad=False)
        self.els = Parameter(Tensor(variable["els"]), name="els", requires_grad=False)
        self.eln = Parameter(Tensor(variable["eln"]), name="eln", requires_grad=False)
        self.ele = Parameter(Tensor(variable["ele"]), name="ele", requires_grad=False)
        self.elw = Parameter(Tensor(variable["elw"]), name="elw", requires_grad=False)
        self.ssurf = Parameter(Tensor(variable["ssurf"]), name="ssurf", requires_grad=False)
        self.tsurf = Parameter(Tensor(variable["tsurf"]), name="tsurf", requires_grad=False)
        self.tbe = Parameter(Tensor(variable["tbe"]), name="tbe", requires_grad=False)
        self.sbe = Parameter(Tensor(variable["sbe"]), name="sbe", requires_grad=False)
        self.sbw = Parameter(Tensor(variable["sbw"]), name="sbw", requires_grad=False)
        self.tbw = Parameter(Tensor(variable["tbw"]), name="tbw", requires_grad=False)
        self.tbn = Parameter(Tensor(variable["tbn"]), name="tbn", requires_grad=False)
        self.tbs = Parameter(Tensor(variable["tbs"]), name="tbs", requires_grad=False)
        self.sbn = Parameter(Tensor(variable["sbn"]), name="sbn", requires_grad=False)
        self.sbs = Parameter(Tensor(variable["sbs"]), name="sbs", requires_grad=False)
        self.wtsurf = Parameter(Tensor(variable["wtsurf"]), name="wtsurf", requires_grad=False)
        self.swrad = Parameter(Tensor(variable["swrad"]), name="swrad", requires_grad=False)
        self.z1 = Parameter(Tensor(variable["z"][0, 0, :]), name="z1", requires_grad=False)
        self.dz1 = Parameter(Tensor(variable["dz"][0:1, 0:1, :]), name="dz1", requires_grad=False)
        self.dzz1 = Parameter(Tensor(variable["dzz"][0:1, 0:1, :]), name="dzz1", requires_grad=False)
        self.z_3d = Parameter(Tensor(np.tile(variable["z"], [self.im, self.jm, 1])), name="z_3d", requires_grad=False)
        self.swrad0 = Parameter(Tensor(np.zeros([self.im, self.jm, 1], np.float32)), name="swrad0", requires_grad=False)
        self.wssurf = Parameter(Tensor(np.zeros([self.im, self.jm, 1], np.float32)), name="wssurf", requires_grad=False)
        self.global_step = Parameter(initializer(0, [1], mstype.int32), name='global_step', requires_grad=False)
        self.q2b = Parameter(self.q2b_init, name="q2b", requires_grad=False)
        self.q2lb = Parameter(self.q2lb_init, name="q2lb", requires_grad=False)
        self.aam = Parameter(self.aam_init, name="aam", requires_grad=False)
        self.cbc = Parameter(self.cbc_init, name="cbc", requires_grad=False)
        self.rmean = Parameter(self.rmean_init, name="rmean", requires_grad=False)
        self.elf = Parameter(Tensor(np.zeros([self.im, self.jm, 1], np.float32)), name="elf", requires_grad=False)

    def dens(self, si, ti, zz, h, fsm):
        """
        density compute function
        Args:
            si: Salinity.
            ti: Potential temperature.
            zz: sigma coordinate, intermediate between Z.
            h: the bottom depth (m)
            fsm: Mask for scalar variables; = 0 over land; = 1 over water
        Returns:
            rhoo[Tensor], density.
        """
        tr = ti + self.tbias
        sr = si + self.sbias
        tr2 = tr * tr
        tr3 = tr2 * tr
        tr4 = tr3 * tr
        p = self.grav * self.rhoref * (-1 * zz * self.mat_ones * h) * 1e-5
        rhor2 = -0.157406e0 + 6.793952e-2 * tr - 9.095290e-3 * tr2 + 1.001685e-4 * tr3 - 1.120083e-6 * tr4 + \
                6.536332e-9 * tr4 * tr
        rhor1 = rhor2 + (0.824493e0 - 4.0899e-3 * tr + 7.6438e-5 * tr2 - 8.2467e-7 * tr3 + 5.3875e-9 * tr4) * sr + \
                (-5.72466e-3 + 1.0227e-4 * tr - 1.6546e-6 * tr2) * self.abs(sr) ** 1.5e0 + 4.8314e-4 * sr * sr
        cr1 = 1449.1e0 + .0821e0 * p + 4.55e0 * tr - .045e0 * tr2 + 1.34e0 * (sr - 35.e0)
        cr = p / (cr1 * cr1)
        rhor = rhor1 + 1.e5 * cr * (1.e0 - 2.e0 * cr)
        rhoo = rhor / self.rhoref * fsm
        return rhoo

    def advct(self, dx, dy, u, v, dt, aam, ub, x_d, y_d, vb):
        """
        compute adv in x, y direction
        Args:
            dx, dy: Increment in x, y direction, respectively.
            u, v: Velocity in x, y direction, respectively.
            ub, vb: Velocity boundary in x, y direction, respectively.
            dt: time step.
            x_d, y_d: Grid increment in x, y direction.
        Returns:
            advx[Tensor], advy[Tensor]
        """
        curv = (self.AYF(v) * self.DXB(self.AXF(dy)) - self.AXF(u) * self.DYB(self.AYF(dx))) / (dx * dy)
        tmp23 = self.AXF(self.AXB(dt) * u) * self.AXF(u) - (dt * aam * 2e0 * self.DXF(ub) / x_d[3])
        tmp23 = tmp23 * (1.0 - self.x_he)
        advx = self.DXB(tmp23) / x_d[2] + self.DYF(self.AXB(self.AYB(dt) * v) * self.AYB(u) - self.AYB(self.AXB(dt))
                                                   * self.AYB(self.AXB(aam)) * (self.DYB(ub) / y_d[0] + self.DXB(vb)
                                                                                / x_d[0])) / y_d[2] - \
               self.AXB(curv * dt * self.AYF(v))
        advx = advx * (1 - self.x_h) * (1 - self.y_he)
        tmp = self.AYF(self.AYB(dt) * v) * self.AYF(v) - dt * aam * 2e0 * self.DYF(vb) / y_d[3]
        tmp = tmp * (1 - self.y_he)
        advy = self.DXF(self.AYB(self.AXB(dt) * u) * self.AXB(v) - self.AYB(self.AXB(dt)) * self.AYB(self.AXB(aam))
                        * (self.DYB(ub) / y_d[0] + self.DXB(vb) / x_d[0])) / x_d[1] + self.DYB(tmp) / y_d[1] \
               + self.AYB(curv * dt * self.AXF(u))
        advy = advy * (1 - self.y_h) * (1 - self.x_he)
        return advx, advy

    def baropg(self, x_d, y_d, z_d, zz, rho, rmean, dt, dz, dum, dvm):
        """
        baropg
        Args:
            x_d, y_d, z_d: Grid increment in x, y, z direction.
            zz: sigma coordinate, intermediate between Z
            rho: density
            rmean: Horizontal mean density field in z-coordinate.
            dt: time step.
            dz: Increment in z direction.
            dum, dvm: Mask for the u, v component of velocity; = 0 over land; =1 over water

        Returns:
            drhox[Tensor], x-component of the internal baroclinic pressure gradient subtract rmean from density before
            integrating
            drhoy[Tensor], y-component of the internal baroclinic pressure gradient subtract rmean from density before
            integrating
        """
        tmp = -1 * self.DZB(zz) * self.DXB(self.AZB(rho - rmean)) / x_d[2] * self.AXB(dt) \
              + self.DXB(dt) / x_d[2] * self.DZB(self.AXB(rho - rmean)) / z_d[1] * self.AZB(zz) * self.AZB(dz)

        tmp1 = -1 * zz * self.AXB(dt) * self.DXB(rho - rmean) / x_d[2]
        tmp = tmp * (1 - self.z_h) + tmp1 * self.z_h
        tmp = tmp * (1 - self.z_e)
        drhox = self.ramp * self.grav * self.AXB(dt) * self.csum(tmp, 2) * dum
        drhox = drhox * (1 - self.z_e)
        tmp = -1 * self.DZB(zz) * self.DYB(self.AZB(rho - rmean)) / y_d[1] * self.AYB(dt) + \
              self.DYB(dt) / y_d[1] * self.DZB(self.AYB(rho - rmean)) / z_d[1] * self.AZB(zz) * self.AZB(dz)
        tmp1 = -1 * zz * self.AYB(dt) * self.DYB(rho - rmean) / y_d[1]
        tmp = tmp * (1 - self.z_h) + tmp1 * self.z_h
        tmp = tmp * (1 - self.z_e)

        drhoy = self.ramp * self.grav * self.AYB(dt) * self.csum(tmp, 2) * dvm
        drhoy = drhoy * (1 - self.z_e)
        return drhox, drhoy

    def lateral_viscosity(self, dx, dy, u, v, dt, aam, ub, vb, x_d, y_d, z_d, rho, rmean):
        """
        lateral viscosity compute function.
        Args:
            dx, dy: Increment in x, y direction, respectively.
            u, v: Velocity in x, y direction, respectively.
            ub, vb: Velocity boundary in x, y direction, respectively.
            x_d, y_d, z_d: Grid increment in x, y, z direction.
            rho: density
            rmean: Horizontal mean density field in z-coordinate.
            dt: time step.

        Returns:
            drhox[Tensor], x-component of the internal baroclinic pressure gradient subtract rmean from density before
            integrating.
            drhoy[Tensor], y-component of the internal baroclinic pressure gradient subtract rmean from density before
            integrating.
            advx[Tensor], advy[Tensor], horizontal advection and diffusion terms.
            amm[Tensor], horizontal kinematic viscosity.
        """
        advx, advy = self.advct(dx, dy, u, v, dt, aam, ub, x_d, y_d, vb)
        drhox, drhoy = self.baropg(x_d, y_d, z_d, self.zz, rho, rmean, dt, self.dz, self.dum, self.dvm)
        aam = self.horcon * dx * dy * self.sqrt(
            self.DXF(u) / x_d[3] * self.DXF(u) / x_d[3] + self.DYF(v) / y_d[3] * self.DYF(v) / y_d[3] +
            0.5 * (self.DYB(self.AYF(self.AXF(u))) / y_d[3] + self.DXB(self.AXF(self.AYF(v))) / x_d[3]) *
            (self.DYB(self.AYF(self.AXF(u))) / y_d[3] + self.DXB(self.AXF(self.AYF(v))) / x_d[3]))

        aam = aam * (1 - self.x_he3d) + self.aam_init * self.x_he3d
        aam = aam * (1 - self.y_he3d) + self.aam_init * self.y_he3d
        aam = aam * (1 - self.z_e) + self.aam_init * self.z_e
        return advx, advy, drhox, drhoy, aam

    def advave(self, x_d, y_d, d, aam2d, uab, vab, ua, va):
        """
        advave
        Args:
            x_d, y_d: Grid increment in x, y direction.
            d: Fluid column depth.
            aam2d: vertical average of aam(m2s-1)
            ua, va: Vertical average velocity in x, y direction, respectively.
            uab, vab: Vertical average velocity boundary in x, y direction, respectively.

        Returns:
            advua[Tensor], advva[Tensor]
        """
        tps = self.AYB(self.AXB(d)) * self.AXB(self.AYB(aam2d)) * (self.DYB(uab) / y_d[0] + self.DXB(vab) / x_d[0])
        tmp = self.AXF(self.AXB(d) * ua) * self.AXF(ua) - 2. * d * aam2d * \
              self.DXF(uab) / x_d[3]
        tmp = tmp * (1 - self.x_he)
        advua = self.DXB(tmp) / x_d[2] + self.DYF(self.AXB(self.AYB(d) * va) * self.AYB(ua) - tps) / y_d[2]
        advua = advua * (1 - self.x_h)
        advua = advua * (1 - self.y_he)

        tmp = self.AYF(self.AYB(d) * va) * self.AYF(va) - 2. * d * aam2d * self.DYF(vab) / y_d[3]
        tmp = tmp * (1 - self.y_he)
        advva = self.DXF(self.AYB(self.AXB(d) * ua) * self.AXB(va) - tps) / x_d[1] + self.DYB(tmp) / y_d[1]
        advva = advva * (1 - self.y_h)
        advva = advva * (1 - self.x_he)
        return advua, advva

    def mode_interaction(self, advx, advy, drhox, drhoy, aam, x_d, y_d, d, uab, vab, ua, va, el):
        """
        external and internal mode variables interaction
        Args:
            drhox, drhoy: x-component and y-component of the internal baroclinic pressure gradient subtract rmean from
            density before integrating.
            aam: horizontal kinematic viscosity.
            x_d, y_d: Grid increment in x, y direction.
            d: Fluid column depth.
            aam2d: vertical average of aam(m2s-1)
            ua, va: Vertical average velocity in x, y direction, respectively.
            uab, vab: Vertical average velocity boundary in x, y direction, respectively.
            el: the surface elevation as used in the external mode (m).

        Returns:
            tuple[Tensor], update varibles of external mode
        """
        adx2d = self.reduce_sum(advx * self.dz, 2)
        ady2d = self.reduce_sum(advy * self.dz, 2)
        drx2d = self.reduce_sum(drhox * self.dz, 2)
        dry2d = self.reduce_sum(drhoy * self.dz, 2)
        aam2d = self.reduce_sum(aam * self.dz, 2)
        advua, advva = self.advave(x_d, y_d, d, aam2d, uab, vab, ua, va)

        adx2d = adx2d - advua
        ady2d = ady2d - advva
        egf = el * self.ispi
        utf = ua * 2.0 * self.AXB(d) * self.isp2i
        vtf = va * 2.0 * self.AYB(d) * self.isp2i
        return adx2d, ady2d, drx2d, dry2d, aam2d, advua, advva, egf, utf, vtf

    def external_el(self, x_d, y_d, d, ua, va, elb):
        """
        compute surface elevation of external 2d mode
        Args:
            x_d, y_d: Grid increment in x, y direction.
            d: Fluid column depth.
            ua, va: Vertical average velocity in x, y direction, respectively.
            elb: the surface elevation boundary as used in the external mode (m).

        Returns:
            elf[Tensor], the surface elevation as used in the external mode (m).
        """
        elf = elb - self.matrix2_dte * ((self.DXF(self.AXB(d) * ua) / x_d[3] + self.DYF(self.AYB(d) * va) / y_d[3])
                                        - self.vfluxf)
        tmp = P.Pad(((0, 1), (0, 0), (0, 0)))(elf)
        tmp = P.Slice()(tmp, (1, 0, 0), P.Shape()(elf))
        elf = elf * (1 - self.x_h) + tmp * self.x_h
        tmp = P.Pad(((1, 0), (0, 0), (0, 0)))(elf)
        tmp = P.Slice()(tmp, (0, 0, 0), P.Shape()(elf))
        elf = elf * (1 - self.x_e) + tmp * self.x_e
        tmp = P.Pad(((0, 0), (0, 1), (0, 0)))(elf)
        tmp = P.Slice()(tmp, (0, 1, 0), P.Shape()(elf))
        elf = elf * (1 - self.y_h) + tmp * self.y_h
        tmp = P.Pad(((0, 0), (1, 0), (0, 0)))(elf)
        tmp = P.Slice()(tmp, (0, 0, 0), P.Shape()(elf))
        elf = elf * (1 - self.y_e) + tmp * self.y_e
        elf = elf * self.fsm
        return elf

    def external_ua(self, iext, x_d, y_d, elf, d, ua, va, uab, vab, el, elb, advua, aam2d, adx2d, drx2d, wubot):
        """
        compute ua of external 2d mode
        Args:
            iext: step of external mode loop.
            x_d, y_d: Grid increment in x, y direction.
            elf: the surface elevation as used in the external mode (m).
            d: Fluid column depth.
            ua, va: Vertical average velocity in x, y direction, respectively.
            uab, vab: Vertical average velocity boundary in x, y direction, respectively.
            el: the surface elevation as used in the external mode (m).
            elb: the surface elevation boundary as used in the external mode (m).
            aam2d: vertical average of aam(m2s-1)
            adx2d: vertical integrals of advx.
            drx2d: vertical integrals of drhox.
            wubot: <wu(-1)> momentum fluxes at the bottom (m2s-2)

        Returns:
            uaf[Tensor], Vertical average velocity in x direction
        """
        if iext % self.ispadv == 0:
            tmp = self.AXF(self.AXB(d) * ua) * self.AXF(ua) - self.mat_twos_2d * d * aam2d * self.DXF(uab) / x_d[3]
            tmp = tmp * (1 - self.x_he)
            advua = self.DXB(tmp) / x_d[2] + self.DYF(self.AXB(self.AYB(d) * va) * self.AYB(ua) - self.AYB(self.AXB(d))
                                                      * self.AXB(self.AYB(aam2d)) * (self.DYB(uab) / y_d[0]
                                                                                     + self.DXB(vab) / x_d[0])) / y_d[2]
            advua = advua * (1 - self.x_h)
            advua = advua * (1 - self.y_he)

        uaf = (self.AXB(self.h + elb) * uab - self.matrix2_dte * (adx2d + advua - self.AXB(self.cor * d * self.AYF(va))
                                                                  + self.matrix2d_grav * self.AXB(d) *
                                                                  ((self.mat_ones_2d - 2.0 * self.alpha) * self.DXB(el)
                                                                   / x_d[2] + self.matrix_alpha
                                                                   * (self.DXB(elb) / x_d[2] + self.DXB(elf) / x_d[2])
                                                                   + self.DXB(self.e_atmos) / x_d[2]) + drx2d
                                                                  + (self.wusurf - wubot))) / self.AXB(self.h + elf)

        tmpua = self.mat_zeros_im_jm_1
        tmpel = self.mat_zeros_im_jm_1
        tmpua = tmpua * (1 - self.x_e1) + self.uabe * self.x_e1
        tmpua = tmpua * (1 - self.x_h1) + self.uabw * self.x_h1
        tmpel = tmpel * (1 - self.x_e1) + self.ele * self.x_e1
        tmpel = tmpel * (1 - self.x_h1) + self.elw * self.x_h1
        tmp = self.matrix_ramp * (tmpua + self.rfe * P.Sqrt()(self.grav / self.h) * (el - tmpel))
        tmp0 = P.Pad(((1, 0), (0, 0), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmp))
        uaf = uaf * (1 - self.x_e) + tmp0 * self.x_e
        tmp = self.matrix_ramp * (tmpua - self.rfw * P.Sqrt()(self.grav / self.h) * (el - tmpel))
        uaf = uaf * (1 - self.x_h1) + tmp * self.x_h1
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(tmp))
        uaf = uaf * (1 - self.x_h) + tmp0 * self.x_h
        uaf = uaf * (1 - self.y_he)
        uaf = uaf * self.dum
        return advua, uaf

    def external_va(self, iext, x_d, y_d, elf, d, ua, va, uab, vab, el, elb, advva, aam2d, ady2d, dry2d, wvbot):
        """
        compute va of external 2d mode
        Args:
            iext: step of external mode loop.
            x_d, y_d: Grid increment in x, y direction.
            elf: the surface elevation as used in the external mode (m).
            d: Fluid column depth.
            ua, va: Vertical average velocity in x, y direction, respectively.
            uab, vab: Vertical average velocity boundary in x, y direction, respectively.
            el: the surface elevation as used in the external mode (m).
            elb: the surface elevation boundary as used in the external mode (m).
            aam2d: vertical average of aam(m2s-1)
            ady2d: vertical integrals of advy.
            dry2d: vertical integrals of drhoy.
            wvbot: <wv(-1)> momentum fluxes at the bottom (m2s-2)

        Returns:
            vaf[Tensor], Vertical average velocity in y direction
        """
        if iext % self.ispadv == 0:
            tmp = self.AYF(self.AYB(d) * va) * self.AYF(va) - self.mat_twos_2d * d * aam2d * self.DYF(vab) / y_d[3]
            tmp = tmp * (1 - self.y_he)
            advva = self.DXF(self.AYB(self.AXB(d) * ua) * self.AXB(va) - self.AYB(self.AXB(d))
                             * self.AXB(self.AYB(aam2d)) * (self.DYB(uab) / y_d[0] +
                                                            self.DXB(vab) / x_d[0])) / x_d[1] + self.DYB(tmp) / y_d[1]
            advva = advva * (1 - self.y_h)
            advva = advva * (1 - self.x_he)

        vaf = (self.AYB(self.h + elb) * vab - self.matrix2_dte * (ady2d + advva + self.AYB(self.cor * d * self.AXF(ua))
                                                                  + self.matrix2d_grav * self.AYB(d) *
                                                                  ((self.mat_ones_2d - 2.0 * self.alpha) *
                                                                   self.DYB(el) / y_d[1] + self.matrix_alpha *
                                                                   (self.DYB(elb) / y_d[1] + self.DYB(elf) / y_d[1]) +
                                                                   self.DYB(self.e_atmos) / y_d[1]) + dry2d +
                                                                  (self.wvsurf - wvbot))) / self.AYB(self.h + elf)

        tmpva = self.mat_zeros_im_jm_1
        tmpel = self.mat_zeros_im_jm_1
        tmpva = tmpva * (1 - self.y_e1) + self.vabn * self.y_e1
        tmpva = tmpva * (1 - self.y_h1) + self.vabs * self.y_h1
        tmpel = tmpel * (1 - self.y_e1) + self.eln * self.y_e1
        tmpel = tmpel * (1 - self.y_h1) + self.els * self.y_h1
        tmp = self.matrix_ramp * (tmpva + self.rfn * P.Sqrt()(self.grav / self.h) * (el - tmpel))
        tmp0 = P.Pad(((0, 0), (1, 0), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmp))
        vaf = vaf * (1 - self.y_e) + tmp0 * self.y_e
        tmp = self.matrix_ramp * (tmpva - self.rfs * P.Sqrt()(self.grav / self.h) * (el - tmpel))
        vaf = vaf * (1 - self.y_h1) + tmp * self.y_h1
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (0, 1, 0), P.Shape()(tmp))
        vaf = vaf * (1 - self.y_h) + tmp0 * self.y_h
        vaf = vaf * (1 - self.x_he)
        vaf = vaf * self.dvm
        return advva, vaf

    def external_update(self, iext, etf, ua, uab, va, vab, el, elb, elf, uaf, vaf, egf, utf, vtf):
        """
        update variabls of external 2d mode
        Args:
            iext: step of external mode loop.
            ua, va: Vertical average velocity in x, y direction, respectively.
            uab, vab: Vertical average velocity boundary in x, y direction, respectively.
            egf: the surface elevation also used in the internal mode for the pressure gradient and derived from el
            utf, vtf: ua, va time averaged over the interval, DT = dti(ms-1)

        Returns:
            tuple[Tensor], update varibles of external mode
        """
        vamax = P.ReduceMax()(P.Abs()(vaf))
        if iext == (self.isplit - 2):
            etf = 0.25 * self.smoth * elf
        elif iext == (self.isplit - 1):
            etf = etf + 0.5 * (1.0 - 0.5 * self.smoth) * elf
        elif iext == self.isplit:
            etf = (etf + 0.5 * elf) * self.fsm

        # TODO fix control flow
        uab = ua + self.matrix_smoth * (uab - self.mat_twos_2d * ua + uaf)
        ua = F.depend(uaf, uab)
        vab = va + self.matrix_smoth * (vab - self.mat_twos_2d * va + vaf)
        va = F.depend(vaf, vab)
        elb = el + self.matrix_smoth * (elb - self.mat_twos_2d * el + elf)
        el = F.depend(elf, elb)
        d = self.h + el
        if iext != self.isplit:
            egf = egf + el * self.matrix_ispi
            utf = utf + self.mat_twos_2d * ua * self.AXB(d) * self.matrix_isp2i
            vtf = vtf + self.mat_twos_2d * va * self.AYB(d) * self.matrix_isp2i
        return etf, uab, ua, vab, va, elb, el, d, egf, utf, vtf, vamax

    def internal_w(self, x_d, y_d, dt, u, v, etf, etb, vfluxb):
        """
        compute velocity in z direction
        Args:
            x_d, y_d: Grid increment in x, y direction.
            dt: time step.
            u, v: Velocity in x, y direction, respectively.
            etf:
            etb:
            vfluxb:

        Returns:
            w[Tensor], velocity in z direction
        """
        del_w = Shift((0, 0, 1))(P.CumSum()(self.dz * (self.DXF(self.AXB(dt) * u) / x_d[3] + self.DYF(self.AYB(dt) * v)
                                                       / y_d[3] + (etf - etb) / self.dti2), 2))
        w = (0.5 * (vfluxb + self.vfluxf) + del_w)
        w_shape = P.Shape()(w)
        tmp_fsm = P.BroadcastTo(w_shape)(self.fsm)
        w = w * tmp_fsm
        w = w * (1. - self.x_he3d)
        w = w * (1. - self.z_e)
        return w

    def adjust_uv(self, u, v, utb, vtb, utf, vtf, dt):
        """
        adjust velocity in x, y direction
        Args:
            u, v: Velocity in x, y direction, respectively.
            utb, vtb: ua, va boundary time averaged over the interval, DT = dti(ms-1)
            utf, vtf: ua, va time averaged over the interval, DT = dti(ms-1)
            dt: time step.

        Returns:
            tuple[Tensor], velocity in x, y direction

        """
        u = u - self.reduce_sum(u * self.dz, 2) + (utb + utf) / (2.0 * self.AXB(dt))
        u = u * (1 - self.x_h3d)
        u = u * (1. - self.z_e)
        v = v - self.reduce_sum(v * self.dz, 2) + (vtb + vtf) / (2.0 * self.AYB(dt))
        v = v * (1 - self.y_h3d)
        v = v * (1 - self.z_e)
        return u, v

    def internal_q(self, x_d, y_d, z_d, etf, aam, q2b, q2lb, q2, q2l, kq, km, kh, u, v, w, dt, dhb, rho,
                   wubot, wvbot, t, s):
        """
        Compute turbulence kinetic energy of internal mode. Refer to the paper "OpenArray v1.0: a simple operator
        library for the decoupling of ocean modeling and parallel computing. " Appendix B formula B13 and B14.
        Args:
            x_d, y_d, z_d: Grid increment in x, y, z direction, respectively.
            q2: Turbulence kinetic energy.
            q2l: Production of turbulence kinetic energy and turbulence length scale.
            km: Vertical kinematic viscosity.
            kh: Vertical mixing coefficient of heat and salinity.
            kq: Vertical mixing coefficient of turbulence kinetic energy.
            u, v, w: Velocity in x, y, w direction, respectively.
            dt: time step.
            t: Potential temperature.
            s: Salinity.

        Returns:
            tuple[Tensor], variables of turbulence kinetic energy.
        """
        dhf = self.h + etf
        q2f = (q2b * dhb - self.dti2 * (-self.DZB(self.AZF(w * q2)) / z_d[1] + self.DXF(self.AXB(q2) * self.AXB(dt) *
                                                                                        self.AZB(u) -
                                                                                        self.AZB(self.AXB(aam)) *
                                                                                        self.AXB(self.h) *
                                                                                        self.DXB(q2b) / x_d[2] *
                                                                                        self.dum) / x_d[3] +
                                        self.DYF(self.AYB(q2) * self.AYB(dt) * self.AZB(v) - self.AZB(self.AYB(aam)) *
                                                 self.AYB(self.h) * self.DYB(q2b) / y_d[1] * self.dvm) / y_d[3])) / dhf
        q2lf = (q2lb * dhb - self.dti2 * (-self.DZB(self.AZF(w * q2l)) / z_d[1] + self.DXF(self.AXB(q2l) * self.AXB(dt)
                                                                                           * self.AZB(u) -
                                                                                           self.AZB(self.AXB(aam)) *
                                                                                           self.AXB(self.h) *
                                                                                           self.DXB(q2lb) / x_d[2] *
                                                                                           self.dum) / x_d[3] +
                                          self.DYF(self.AYB(q2l) * self.AYB(dt) * self.AZB(v) - self.AZB(self.AYB(aam))
                                                   * self.AYB(self.h) * self.DYB(q2lb) / y_d[1] * self.dvm) /
                                          y_d[3])) / dhf

        a1 = 0.92
        b1 = 16.6
        a2 = 0.74
        b2 = 10.1
        c1 = 0.08
        e1 = 1.8
        e2 = 1.33
        sef = 1.0
        cbcnst = 100.0
        surfl = 2.e5
        shiw = 0.0

        dzz1 = Shift((0, 0, 1))(self.dzz1)
        a = self.dz1 * dzz1 * self.mat_ones
        a = - self.dti2 * (self.AZF(kq) + self.umol) / (a * dhf * dhf + self.small_debug)
        a = a * (1 - self.z_he)

        dz1 = Shift((0, 0, 1))(self.dz1)
        c = dz1 * dzz1 * self.mat_ones
        c = - self.dti2 * (self.AZB(kq) + self.umol) / (c * dhf * dhf + self.small_debug)
        c = c * (1 - self.z_he)

        utau2 = P.Sqrt()(self.AXF(self.wusurf) * self.AXF(self.wusurf) + self.AYF(self.wvsurf) * self.AYF(self.wvsurf))

        ee = self.mat_zeros
        gg = (15.8 * cbcnst) ** (2.0 / 3.0) * utau2 * self.z_h * self.mat_ones
        l0 = surfl * utau2 / self.grav * self.mat_ones

        tmp = P.Sqrt()(self.AXF(wubot) * self.AXF(wubot) + self.AYF(wvbot) * self.AYF(wvbot)) * (16.6 ** (2.0 / 3.0)) \
              * sef
        q2f = q2f * (1 - self.z_e) + tmp * self.z_e
        p = self.grav * self.rhoref * (-self.zz * self.mat_ones * self.h) * 1.e-4
        cc = 1449.10 + 0.00821 * p + 4.55 * (t + self.tbias) - 0.045 * (t + self.tbias) * (t + self.tbias) + 1.34 * \
             (s + self.sbias - 35.0)
        cc = cc / P.Sqrt()((1.0 - 0.01642 * p / cc) * (1.0 - 0.4 * p / (cc * cc)))
        cc = cc * (1 - self.z_e)

        q2b = P.Abs()(q2b)
        q2lb = P.Abs()(q2lb)
        boygr = self.matrix_grav * (self.matrix_grav * self.h - self.DZB(rho) * self.AZB(cc * cc) / z_d[1]) / \
                (self.h * self.AZB(cc * cc))
        boygr = boygr * (1 - self.z_h)
        l_tmp = q2lb / q2b
        kappa_l0 = self.kappa * l0

        tmp = P.Cast()(kappa_l0 > l_tmp, mstype.float32) * P.Cast()(self.z_3d > -0.5, mstype.float32)
        l_tmp = kappa_l0 * tmp + l_tmp * (1 - tmp)
        gh = l_tmp * l_tmp * boygr / q2b
        tmp = gh < 0.028
        tmp = P.Cast()(tmp, mstype.float32)
        gh = gh * tmp + 0.028 * (1 - tmp)
        l_tmp = l_tmp * (1 - self.z_h) + self.kappa * l0 * self.z_h
        l_tmp = l_tmp * (1 - self.z_e)
        gh = gh * (1 - self.z_he)
        kn = km * sef * (self.DZB(self.AXF(u)) / z_d[1] * self.DZB(self.AXF(u)) / z_d[1] + self.DZB(self.AYF(v)) / z_d[
            1] * self.DZB(self.AYF(v)) / z_d[1]) / (dhf * dhf) - shiw * km * boygr + kh * boygr
        kn = kn * (1 - self.x_e3d)

        dtef = P.Sqrt()(q2b) / (b1 * l_tmp + self.small)
        dtef = dtef * (1 - self.z_he)

        ggtmp = gg
        eetmp = ee
        for k in range(1, self.kbm1):
            gg = 1.0 / (a + c * (1 - Shift((0, 0, 1))(eetmp)) - self.matrix2_dti2 * dtef - 1.0)
            ee = a * gg
            gg = (-self.matrix2_dti2 * kn + c * Shift((0, 0, 1))(ggtmp) - q2f) * gg
            ee = eetmp * (1 - self.zslice[k]) + ee * self.zslice[k]
            gg = ggtmp * (1 - self.zslice[k]) + gg * self.zslice[k]
            ggtmp = gg
            eetmp = ee

        q2f = q2f * (1 - self.x_he3d)
        q2ftmp = q2f
        for k in range(self.kbm1 - 1, -1, -1):
            q2f = ee * Shift((0, 0, -1))(q2f) + gg
            q2f = q2ftmp * (1 - self.zslice[k]) + q2f * self.zslice[k]
            q2ftmp = q2f
        q2f = q2f * (1 - self.y_e3d)
        ggtmp = gg
        eetmp = ee

        ee = ee * (1 - self.z_h1)
        gg = gg * (1 - self.z_h1)
        q2lf = q2lf * (1 - self.z_he)

        dtef = dtef * (1.0 + e2 * ((1.0 / P.Abs()(
            self.z1 - P.Slice()(self.z1, (0,), (1,)) + self.small_debug) +
                                    1.0 / P.Abs()(self.z1 - P.Slice()(self.z1, (self.kb - 1,), (1,)) +
                                                  self.small_debug)) * l_tmp / (dhf * self.kappa)) *
                       ((1.0 / P.Abs()(self.z1 - P.Slice()(self.z1, (0,), (1,)) + self.small_debug) +
                         1.0 / P.Abs()(self.z1 - P.Slice()(self.z1, (self.kb - 1,), (1,)) + self.small_debug)) *
                        l_tmp / (dhf * self.kappa)))

        for k in range(1, self.kbm1):
            gg = 1.0 / (a + c * (1 - Shift((0, 0, 1))(eetmp)) - self.matrix1_dti2 * dtef - 1.0)
            ee = a * gg
            gg = (-self.matrix1_dti2 * kn * l_tmp * e1 + c * Shift((0, 0, 1))(ggtmp) - q2lf) * gg
            ee = eetmp * (1 - self.zslice[k]) + ee * self.zslice[k]
            gg = ggtmp * (1 - self.zslice[k]) + gg * self.zslice[k]
            ggtmp = gg
            eetmp = ee
        gg = gg * (1 - self.x_e3d)

        q2lftmp = q2lf
        for k in range(self.kbm1 - 1, 0, -1):
            q2lf = ee * Shift((0, 0, -1))(q2lf) + gg
            q2lf = q2lftmp * (1 - self.zslice[k]) + q2lf * self.zslice[k]
            q2lftmp = q2lf

        res_compare = P.LogicalOr()((q2f <= self.small), (q2lf <= self.small))
        filters = P.Cast()(res_compare, mstype.float32)
        filters = filters * (1 - self.z_he)
        q2f = q2f * (1 - filters) + self.matrix_small * filters
        q2lf = q2lf * (1 - filters) + 0.1 * dt * self.small * filters

        tmp = -gh * (3.0 * a2 * b2 + 18.0 * a1 * a2) + 1
        sh = a2 * (1.0 - 6.0 * a1 / b1) / tmp
        tmp = - gh * (9.0 * a1 * a2) + 1
        sm = (a1 * (1.0 - 3.0 * c1 - 6.0 * a1 / b1) + sh * (18.0 * a1 * a1 + 9.0 * a1 * a2) * gh) / tmp

        kn = l_tmp * P.Sqrt()(P.Abs()(q2))
        kq = (kn * sh * 0.41 + kq) * 0.5
        km = (kn * sm + km) * 0.5
        kh = (kn * sh + kh) * 0.5

        tmp0 = P.Pad(((0, 0), (1, 0), (0, 0)))(km)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(km))
        km = km * (1 - self.y_e) + tmp0 * self.fsm * self.y_e
        tmp0 = P.Pad(((0, 0), (1, 0), (0, 0)))(kh)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(kh))
        kh = kh * (1 - self.y_e) + tmp0 * self.fsm * self.y_e
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(km)
        tmp0 = P.Slice()(tmp0, (0, 1, 0), P.Shape()(km))
        km = km * (1 - self.y_h) + tmp0 * self.fsm * self.y_h
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(kh)
        tmp0 = P.Slice()(tmp0, (0, 1, 0), P.Shape()(kh))
        kh = kh * (1 - self.y_h) + tmp0 * self.fsm * self.y_h

        tmp0 = P.Pad(((1, 0), (0, 0), (0, 0)))(km)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(km))
        km = km * (1 - self.x_e) + tmp0 * self.fsm * self.x_e
        tmp0 = P.Pad(((1, 0), (0, 0), (0, 0)))(kh)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(kh))
        kh = kh * (1 - self.x_e) + tmp0 * self.fsm * self.x_e
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(km)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(km))
        km = km * (1 - self.x_h) + tmp0 * self.fsm * self.x_h
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(kh)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(kh))
        kh = kh * (1 - self.x_h) + tmp0 * self.fsm * self.x_h

        tmpu = u
        tmpv = v
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(tmpu)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(tmpu))
        tmpu = tmpu * (1 - self.x_h) + tmp0 * self.x_h
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(tmpu)
        tmp0 = P.Slice()(tmp0, (0, 1, 0), P.Shape()(tmpu))
        tmpu = tmpu * (1 - self.y_h) + tmp0 * self.y_h

        # ==========================EAST
        tmp1 = q2 - self.dti * ((0.5 * (u - P.Abs()(u))) * (self.small - q2) / self.AXB(self.dx) +
                                (0.5 * (u + P.Abs()(u))) * self.DXB(q2) / x_d[2])
        q2f = q2f * (1 - self.x_e) + tmp1 * self.x_e
        tmp1 = q2l - self.dti * ((0.5 * (u - P.Abs()(u))) * (self.small - q2l) / self.AXB(self.dx) +
                                 (0.5 * (u + P.Abs()(u))) * self.DXB(q2l) / x_d[2])
        q2lf = q2lf * (1 - self.x_e) + tmp1 * self.x_e

        # ==========================WEST
        tmp1 = q2 - self.dti * ((0.5 * (tmpu + P.Abs()(tmpu))) * (q2 - self.small) / self.AXF(self.dx) +
                                (0.5 * (tmpu - P.Abs()(tmpu))) * self.DXF(q2) / x_d[2])
        q2f = q2f * (1 - self.x_h) + tmp1 * self.x_h
        tmp1 = q2l - self.dti * ((0.5 * (tmpu + P.Abs()(tmpu))) * (q2l - self.small) / self.AXF(self.dx) +
                                 (0.5 * (tmpu - P.Abs()(tmpu))) * self.DXF(q2l) / x_d[2])
        q2lf = q2lf * (1 - self.x_h) + tmp1 * self.x_h

        # ==========================NORTH
        tmp1 = q2 - self.dti * ((0.5 * (v - P.Abs()(v))) * (self.small - q2) / self.AYB(self.dy) +
                                (0.5 * (v + P.Abs()(v))) * self.DYB(q2) / y_d[1])
        q2f = q2f * (1 - self.y_e) + tmp1 * self.y_e
        tmp1 = q2l - self.dti * ((0.5 * (v - P.Abs()(v))) * (self.small - q2l) / self.AYB(self.dy) +
                                 (0.5 * (v + P.Abs()(v))) * self.DYB(q2l) / y_d[1])
        q2lf = q2lf * (1 - self.y_e) + tmp1 * self.y_e

        # ==========================SOUTH
        tmp1 = q2 - self.dti * ((0.5 * (tmpv + P.Abs()(tmpv))) * (q2 - self.small) / self.AYF(self.dy) +
                                (0.5 * (tmpv - P.Abs()(tmpv))) * self.DYF(q2) / y_d[1])
        q2f = q2f * (1 - self.y_h) + tmp1 * self.y_h
        tmp1 = q2l - self.dti * ((0.5 * (tmpv + P.Abs()(tmpv))) * (q2l - self.small) / self.AYF(self.dy) +
                                 (0.5 * (tmpv - P.Abs()(tmpv))) * self.DYF(q2l) / y_d[1])
        q2lf = q2lf * (1 - self.y_h) + tmp1 * self.y_h

        q2f = q2f * self.fsm + 1.e-10
        q2lf = q2lf * self.fsm + 1.e-10

        q2b = q2 + 0.5 * self.smoth * (q2f + q2b - 2.0 * q2)
        q2 = q2f
        q2lb = q2l + 0.5 * self.smoth * (q2lf + q2lb - 2.0 * q2l)
        q2l = q2lf

        return dhf, a, c, gg, ee, kq, km, kh, q2b, q2, q2lb, q2l

    def internal_u(self, x_d, z_d, dhf, u, v, w, ub, vb, egf, egb, ee, gg, cbc, km, advx, drhox, dt, dhb):
        """
        compute velocity in x direction of internal mode. Refer to the paper "OpenArray v1.0: a simple operator
        library for the decoupling of ocean modeling and parallel computing. " Appendix B formula B9.
        Args:
            x_d, z_d: Grid increment in x, z direction, respectively.
            u, v, w: Velocity in x, y, w direction, respectively.
            ub, vb: Velocity boundary in x, y direction, respectively.
            km: Vertical kinematic viscosity.
            egf: the surface elevation also used in the internal mode for the pressure gradient and derived from el
            egb: the surface elevation boundary also used in the internal mode for the pressure gradient and derived
            from el.
            dt: time step.

        Returns:
            tuple[Tensor], variables of velocity of x direction.
        """
        dh = self.AXB(dhf)
        tmp = self.AXB(w) * self.AZB(u)
        dztmp = z_d[0] * (1 - self.z_e) + self.small_debug * self.z_e
        dzztmp = self.dzz * (1 - self.z_e) + self.small_debug * self.z_e

        uf = (self.AXB(dhb) * ub - self.dti2 * (advx + drhox - self.AXB(self.cor * dt * self.AYF(v)) + self.grav *
                                                self.AXB(dt) * (self.DXB(egf + egb) / x_d[2] + self.DXB(self.e_atmos) /
                                                                x_d[2] * 2.0) * 0.5 - self.DZF(tmp) / dztmp)) / dh
        uf = uf * (1 - self.z_e)
        dh = dh * (1 - self.x_h) + self.x_h
        dh = dh * (1 - self.y_h) + self.y_h

        c = self.AXB(km)
        c = c * (1 - self.x_h3d)
        a = -self.dti2 * (Shift((0, 0, -1))(c) + self.umol)
        a = a / (self.dz * self.dzz * dh * dh + self.small_debug)
        a = a * (1 - self.z_e) * (1 - self.z_e1)
        c = -self.dti2 * (c + self.umol) / (dztmp * Shift((0, 0, 1))(dzztmp) * dh * dh + self.small_debug)
        c = c * (1 - self.z_he)
        ee = ee * (1 - self.z_h) + a / (a - 1.0) * self.z_h
        gg = gg * (1 - self.z_h) + (self.dti2 * self.wusurf / (dztmp * dh) - uf) / (a - 1.0) * self.z_h

        ggtmp = gg
        eetmp = ee
        for k in range(1, self.kbm2):
            gg = 1.0 / (a + c * (1.0 - Shift((0, 0, 1))(eetmp)) - 1.0)
            ee = a * gg
            gg = (c * Shift((0, 0, 1))(ggtmp) - uf) * gg
            ee = eetmp * (1 - self.zslice[k]) + ee * self.zslice[k]
            gg = ggtmp * (1 - self.zslice[k]) + gg * self.zslice[k]
            ggtmp = gg
            eetmp = ee
        tmp = P.Sqrt()(ub * ub + self.AXB(self.AYF(vb)) * self.AXB(self.AYF(vb)))
        tps = self.AXB(cbc) * P.Slice()(tmp, (0, 0, self.kbm1 - 1), (self.im, self.jm, 1))
        uf = uf * (1 - self.z_e1) + (c * Shift((0, 0, 1))(gg) - uf) / (tps * self.dti2 / (-dztmp * dh) - 1.0 - c *
                                                                       (Shift((0, 0, 1))(ee) - 1.0)) * self.z_e1
        uftmp = uf
        for k in range(self.kbm2 - 1, -1, -1):
            uf = ee * Shift((0, 0, -1))(uf) + gg
            uf = uftmp * (1 - self.zslice[k]) + uf * self.zslice[k]
            uftmp = uf

        uf = uf * self.dum
        wubot = P.Slice()(-tps * uf, (0, 0, self.kbm2), (self.im, self.jm, 1))

        tmph = self.h
        tmp0 = P.Pad(((1, 0), (0, 0), (0, 0)))(tmph)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmph))
        tmph = tmph * (1 - self.x_h13d) + tmp0 * self.x_h13d
        tmpu = u
        tmp0 = P.Pad(((1, 0), (0, 0), (0, 0)))(tmpu)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmpu))
        tmpu = tmpu * (1 - self.x_e3d) + tmp0 * self.x_e3d
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(tmpu)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(tmpu))
        tmpu = tmpu * (1 - self.x_h13d) + tmp0 * self.x_h13d

        tmp = P.Sqrt()(tmph / self.hmax) * self.AYF(self.AYB(tmpu)) + (1.0 - P.Sqrt()(tmph / self.hmax)) * self.AYF(
            self.AYB(u))
        uftmp = uf
        uf = uf * (1 - self.x_e3d) + tmp * self.x_e3d
        tmp0 = P.Pad(((0, 1), (0, 0), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (1, 0, 0), P.Shape()(tmp))
        uf = uf * (1 - self.x_h3d) + tmp0 * self.x_h3d
        uf = uf * (1 - self.x_h13d) + tmp * self.x_h13d
        uf = uf * (1 - self.y_he3d)
        uf = uf * self.dum
        uf = uf * (1 - self.z_e) + uftmp * self.dum * self.z_e

        return uf, a, c, gg, ee, wubot

    def internal_v(self, y_d, z_d, dhf, u, v, w, ub, vb, egf, egb, ee, gg, cbc, km, advy, drhoy, dt, dhb):
        """
        compute velocity in y direction of internal mode. Refer to the paper "OpenArray v1.0: a simple operator
        library for the decoupling of ocean modeling and parallel computing. " Appendix B formula B10.
        Args:
            y_d, z_d: Grid increment in y, z direction, respectively.
            u, v, w: Velocity in x, y, w direction, respectively.
            ub, vb: Velocity boundary in x, y direction, respectively.
            egf: the surface elevation also used in the internal mode for the pressure gradient and derived from el
            egb: the surface elevation boundary also used in the internal mode for the pressure gradient and derived
            from el.
            dt: time step.

        Returns:
            tuple[Tensor], variables of velocity of x direction.
        """
        dh = self.AYB(dhf)
        tmp = self.AYB(w) * self.AZB(v)
        dztmp = z_d[0] * (1 - self.z_e) + self.small_debug * self.z_e
        dzztmp = self.dzz * (1 - self.z_e) + self.small_debug * self.z_e

        vf = (self.AYB(dhb) * vb - self.dti2 * (advy + drhoy + self.AYB(self.cor * dt * self.AXF(u)) + self.grav *
                                                self.AYB(dt) * (self.DYB(egf + egb) / y_d[1] + self.DYB(self.e_atmos) /
                                                                y_d[1] * 2.0) / 2.0 - self.DZF(tmp) / dztmp)) / dh
        vf = vf * (1 - self.z_e) * (1 - self.x_e3d)
        dh = dh * (1 - self.x_h) + self.x_h
        dh = dh * (1 - self.y_h) + self.y_h

        c = self.AYB(km)
        c = c * (1 - self.y_h3d)
        a = -self.dti2 * (Shift((0, 0, -1))(c) + self.umol)
        a = a / (dztmp * dzztmp * dh * dh)
        a = a * (1 - self.z_e) * (1 - self.z_e1)
        c = -self.dti2 * (c + self.umol) / (dztmp * Shift((0, 0, 1))(dzztmp) * dh * dh + self.small_debug)
        c = c * (1 - self.z_he)
        ee = ee * (1 - self.z_h) + a / (a - 1.0) * self.z_h
        gg = gg * (1 - self.z_h) + (self.dti2 * self.wvsurf / (dztmp * dh) - vf) / (a - 1.0) * self.z_h

        ggtmp = gg
        eetmp = ee
        for k in range(1, self.kbm2):
            gg = 1.0 / (a + c * (1.0 - Shift((0, 0, 1))(eetmp)) - 1.0)
            ee = a * gg
            gg = (c * Shift((0, 0, 1))(ggtmp) - vf) * gg  # * gg
            ee = eetmp * (1 - self.zslice[k]) + ee * self.zslice[k]
            gg = ggtmp * (1 - self.zslice[k]) + gg * self.zslice[k]
            ggtmp = gg
            eetmp = ee
        tmp = P.Sqrt()(self.AYB(self.AXF(ub)) * self.AYB(self.AXF(ub)) + vb * vb)
        tps = self.AYB(cbc) * P.Slice()(tmp, (0, 0, self.kbm1 - 1), (self.im, self.jm, 1))
        tps = tps * (1 - self.x_e)
        vf = vf * (1 - self.z_e1) + (c * Shift((0, 0, 1))(gg) - vf) / (tps * self.dti2 / (-dztmp * dh) - 1.0 - c *
                                                                       (Shift((0, 0, 1))(ee) - 1.0)) * self.z_e1
        vftmp = vf
        for k in range(self.kbm2 - 1, -1, -1):
            vf = ee * Shift((0, 0, -1))(vf) + gg
            vf = vftmp * (1 - self.zslice[k]) + vf * self.zslice[k]
            vftmp = vf

        vf = vf * self.dvm
        wvbot = P.Slice()(-tps * vf, (0, 0, self.kbm2), (self.im, self.jm, 1))

        tmph = self.h
        tmp0 = P.Pad(((0, 0), (1, 0), (0, 0)))(tmph)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmph))
        tmph = tmph * (1 - self.y_h13d) + tmp0 * self.y_h13d
        tmpv = v
        tmp0 = P.Pad(((0, 0), (1, 0), (0, 0)))(tmpv)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmpv))
        tmpv = tmpv * (1 - self.y_e3d) + tmp0 * self.y_e3d
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(tmpv)
        tmp0 = P.Slice()(tmp0, (0, 0, 0), P.Shape()(tmpv))
        tmpv = tmpv * (1 - self.y_h13d) + tmp0 * self.y_h13d

        tmp = P.Sqrt()(tmph / self.hmax) * self.AXF(self.AXB(tmpv)) + (1.0 - P.Sqrt()(tmph / self.hmax)) * self.AXF(
            self.AXB(v))
        vftmp = vf
        vf = vf * (1 - self.y_e3d) + tmp * self.y_e3d
        tmp0 = P.Pad(((0, 0), (0, 1), (0, 0)))(tmp)
        tmp0 = P.Slice()(tmp0, (0, 1, 0), P.Shape()(tmp))
        vf = vf * (1 - self.y_h3d) + tmp0 * self.y_h3d
        vf = vf * (1 - self.y_h13d) + tmp * self.y_h13d
        vf = vf * (1 - self.x_he3d)
        vf = vf * self.dvm
        vf = vf * (1 - self.z_e) + vftmp * self.dvm * self.z_e

        return vf, a, c, gg, ee, wvbot

    def adjust_ufvf(self, u, v, uf, vf, ub, vb):
        """
        adjust velocity in x, y direction
        Args:
            u, v: Velocity in x, y direction, respectively.
            ub, vb: Velocity boundary in x, y direction, respectively.
            uf, vf: Velocity in x, y direction, respectively.

        Returns:
            tuple[Tensor], velocity and velocity boundary in x, y direction
        """
        u = u + 0.5 * self.smoth * (uf + ub - 2.0 * u - self.reduce_sum((uf + ub - 2.0 * u) * self.dz, 2))
        u = u * (1 - self.z_e)

        v = v + 0.5 * self.smoth * (vf + vb - 2.0 * v - self.reduce_sum((vf + vb - 2.0 * v) * self.dz, 2))
        v = v * (1 - self.z_e)

        ub = u
        u = uf
        vb = v
        v = vf
        return u, v, ub, vb

    def internal_update(self, egf, etb, utf, vtf, etf, et):
        """
        update variablse of internal mode
        Args:
            egf: the surface elevation also used in the internal mode for the pressure gradient and derived from el.
            etb: the surface elevation boundary as used in the internal mode and derived from EL (m).
            utf, vtf: ua, va time averaged over the interval, DT = dti(ms-1).
            et: the surface elevation as used in the internal mode and derived from EL (m).

        Returns:
            tuple[Tensor], variables of internal mode
        """
        egb = egf
        etb = et
        et = etf
        dt = self.h + et
        dhb = etb + self.h
        utb = utf
        vtb = vtf
        vfluxb = self.vfluxf
        return egb, etb, dt, dhb, utb, vtb, vfluxb, et

    def internal_t_(self, f, fb, wfsurf, fsurf, frad, fclim, fbe, fbw, fbn, fbs, x_d, y_d, z_d, dt, u, aam, h, dum, v,
                    dvm, w, dhf, etf, a, kh, dzz, c, dzz1, ee, gg, dx, dz, dy, fsm, dhb):
        """
        Compute Potential temperature and Salinity of internal mode. Refer to the paper "OpenArray v1.0: a simple
        operator library for the decoupling of ocean modeling and parallel computing. " Appendix B formula B11 and B12.
        Args:
            f: Potential temperature or Salinity.
            fb: Potential temperature or Salinity boundary.
            wfsurf: (<w θ (0)>, <ws(0)>) temperature and salinity fluxes at the surface (ms-1 oC, ms-1 psu)
            fclim: Climatology of temperature or salinity.
            fbe, fbw, fbn, fbs: Potential temperature or Salinity boundary in east, west, north, south direction.
            x_d, y_d, z_d: Grid increment in x, y, z direction, respectively.
            kh: Vertical mixing coefficient of heat and salinity.
            u, v, w: Velocity in x, y, w direction, respectively.
            dt: time step.
            aam: horizontal kinematic viscosity.
            dum, dvm: Mask for the u, v component of velocity; = 0 over land; =1 over water
            fsm: Mask for scalar variables; = 0 over land; = 1 over water.

        Returns:
            tuple[Tensor], variables of Potential temperature and Salinity.
        """
        tmpdz = z_d[0] * (1 - self.z_e) + self.small_debug * self.z_e
        tmpdzz = dzz * (1 - self.z_e) + self.small_debug * self.z_e

        tmp = self.AZB(f) * w
        tmp1 = f * w
        tmp = tmp * (1 - self.z_h) + tmp1 * self.z_h
        tmp = tmp * (1 - self.z_e)
        ff = (dhb * fb - self.dti2 * (self.DXF(self.AXB(dt) * self.AXB(f) * u - self.AXB(aam) * self.AYB(h) *
                                               self.tprni * self.DXB(fb) / x_d[2] * dum) / x_d[3] +
                                      self.DYF(self.AYB(dt) * self.AYB(f) * v - self.AYB(aam) * self.AYB(h) *
                                               self.tprni * self.DYB(fb) / y_d[1] * dvm) / y_d[3] - self.DZF(tmp) /
                                      tmpdz)) / dhf
        ff_shape = P.Shape()(ff)
        tmp_ze = P.BroadcastTo(ff_shape)(self.z_e)
        ff = ff * (1 - tmp_ze) * (1 - self.x_he) * (1 - self.y_he)

        rad = self.mat_zeros
        dh = h + etf
        shape_kh = P.Shape()(kh)
        tmp_kh = -1 * self.dti2 * (P.Slice()(kh, (0, 0, 2), (shape_kh[0], shape_kh[1], shape_kh[2] - 2)) + self.umol)
        tmp_kh = P.Pad(((0, 0), (0, 0), (0, 2)))(tmp_kh)
        a = a * (1 - self.z_he2) + tmp_kh * self.z_he2
        a = a / (tmpdz * tmpdzz * dhf * dhf)
        a = a * self.z_he2

        shape_c = P.Shape()(c)
        tmp_c_dzz1 = P.BroadcastTo(shape_c)(dzz1)
        tmp_c_dzz1 = P.Slice()(tmp_c_dzz1, (0, 0, 0), (shape_c[0], shape_c[1], shape_c[2] - 2))
        tmp_c_dzz1 = P.Pad(((0, 0), (0, 0), (1, 1)))(tmp_c_dzz1)

        c = c * self.z_he + tmp_c_dzz1 * (1 - self.z_he)
        c = -1 * self.dti2 * (kh + self.umol) / (tmpdz * c * dhf * dhf)
        # DIV ===> NAN
        c = P.Slice()(c, (0, 0, 1), (shape_c[0], shape_c[1], shape_c[2] - 2))
        c = P.Pad(((0, 0), (0, 0), (1, 1)))(c)

        ee = (a / (a - 1)) * self.z_h + ee * (1 - self.z_h)
        gg = (((self.dti2 * wfsurf) / (tmpdz * dh) - ff) / (a - 1)) * self.z_h + gg * (1 - self.z_h)

        for k in range(1, self.kbm1):
            gg = (1e0 / (a + c * (1 - Shift((0, 0, 1))(ee)) - 1e0)) * self.zslice[k] + gg * (1 - self.zslice[k])
            ee = ee * (1 - self.zslice[k]) + a * gg * self.zslice[k]
            gg = ((c * Shift((0, 0, 1))(gg) - ff + self.dti2 * (rad - Shift((0, 0, -1))(rad)) /
                   (dh * tmpdz)) * gg) * self.zslice[k] + gg * (1 - self.zslice[k])

        fftmp = ((c * Shift((0, 0, 1))(gg) - ff + self.dti2 * (rad - Shift((0, 0, -1))(rad)) / (dh * tmpdz)) /
                 (c * (1 - Shift((0, 0, 1))(ee)) - 1e0))
        zslicetmp = P.BroadcastTo(P.Shape()(fftmp))(self.zslice[self.kbm1 - 1])
        ff = ff * (1 - zslicetmp) + fftmp * zslicetmp

        for k in range(self.kbm2 - 1, -1, -1):
            ff = ff * (1 - self.zslice[k]) + (ee * Shift((0, 0, -1))(ff) + gg) * self.zslice[k]

        # bcond4
        tmptb = self.mat_zeros
        tmptb = fbe * self.x_e + tmptb * (1 - self.x_e)
        tmptb = fbw * self.x_h + tmptb * (1 - self.x_h)
        tmptb = fbn * self.y_e + tmptb * (1 - self.y_e)
        tmptb = fbs * self.y_h + tmptb * (1 - self.y_h)
        tmpu = u
        tmpv = v
        tmpu = Shift((1, 0, 0))(tmpu) * self.x_e1 + tmpu * (1 - self.x_e1)
        tmpu = Shift((1, 0, 0))(tmpu) * self.x_h1 + tmpu * (1 - self.x_h1)
        tmpv = Shift((0, 1, 0))(tmpv) * self.y_e1 + tmpv * (1 - self.y_e1)
        tmpv = Shift((0, 1, 0))(tmpv) * self.y_h1 + tmpv * (1 - self.y_h1)

        # ==========================EAST
        tmp1 = f - self.dti * ((0.5 * (u - P.Abs()(u))) * (tmptb - f) / self.AXB(dx) + (0.5 * (u + P.Abs()(u))) *
                               self.DXB(f) / x_d[2])
        tmp1 = tmp1 * (1 - self.x_h3d)
        ff = ff * (1 - self.x_e3d * (1 - self.z_e)) + tmp1 * self.x_e3d * (1 - self.z_e)
        tmp_test = P.Pad(((0, 1), (0, 0), (0, 0)))(tmp1)
        tmp_test = P.Slice()(tmp_test, (1, 0, 0), P.Shape()(tmp1))
        tmp1 = (1 - self.x_e13d) * tmp1 + tmp_test * self.x_e13d
        tmp2 = tmp1 + 0.5 * self.dti * (tmpu + P.Abs()(tmpu)) / (tmpu + self.small_debug) * self.AZF(w) / dt * \
               self.DZF(self.AZB(f)) / z_d[0] * dz / self.AZB(dzz)
        tmp2_del_nan = P.Slice()(tmp2, (self.imm1 - 1, 0, 1), (1, self.jm, self.kb - 3))
        tmp2 = P.Pad(((self.im - 1, 0), (0, 0), (1, 2)))(tmp2_del_nan)
        ff = ff * (1 - self.x_e3d * self.z_he3) + tmp2 * self.x_e3d * self.z_he3

        # ==========================WEST
        tmp1 = f - self.dti * ((0.5 * (tmpu + P.Abs()(tmpu))) * (f - tmptb) / self.AXF(dx) +
                               (0.5 * (tmpu + P.Abs()(tmpu))) * self.DXF(f) / x_d[2] * dx / self.AXF(dx))
        ff = ff * (1 - self.x_h3d * (1 - self.z_e)) + tmp1 * self.x_h3d * (1 - self.z_e)
        tmp_test = P.Pad(((1, 0), (0, 0), (0, 0)))(tmp1)
        tmp_test = P.Slice()(tmp_test, (0, 0, 0), P.Shape()(tmp1))
        tmp1 = (1 - self.x_h13d) * tmp1 + tmp_test * self.x_h13d
        tmp2 = tmp1 + 0.5 * self.dti * (u + P.Abs()(u)) / u * self.AZF(w) / dt * self.DZF(self.AZB(f)) / self.AZB(dzz)
        tmp2 = P.Pad(((0, self.im - 1), (0, 0), (1, 2)))(tmp2[1:2, :, 1:self.kbm2])
        ff = ff * (1 - self.x_h3d * self.z_he3) + tmp2 * self.x_h3d * self.z_he3

        # ==========================NORTH
        tmp1 = f - self.dti * ((0.5 * (v - P.Abs()(v))) * (tmptb - f) / self.AYB(dy) + (0.5 * (v + P.Abs()(v))) *
                               self.DYB(f) / y_d[1])
        ff = ff * (1 - self.y_e3d * (1 - self.z_e)) + tmp1 * self.y_e3d * (1 - self.z_e)
        tmp1 = Shift((0, -1, 0))(tmp1) * self.y_e13d + tmp1 * (1 - self.y_e13d)
        tmp2 = tmp1 + 0.5 * self.dti * (tmpv + P.Abs()(tmpv)) / tmpv * self.AZF(w) / dt * self.DZF(self.AZB(f)) / z_d[
            0] * dz / self.AZB(dzz)
        tmp2 = P.Pad(((0, 0), (self.jm - 1, 0), (1, 2)))(tmp2[:, self.jmm1 - 1:self.jmm1, 1:self.kbm2])
        ff = ff * (1 - self.y_e3d * self.z_he3) + tmp2 * self.y_e3d * self.z_he3

        # ==========================SOUTH
        tmp1 = f - self.dti * ((0.5 * (tmpv + P.Abs()(tmpv))) * (f - tmptb) / self.AYF(dy) +
                               (0.5 * (tmpv + P.Abs()(tmpv))) * self.DYF(f) / y_d[1])
        ff = ff * (1 - self.y_h3d * (1 - self.z_e)) + tmp1 * self.y_h3d * (1 - self.z_e)
        tmp1 = Shift((0, 1, 0))(tmp1) * self.y_h13d + tmp1 * (1 - self.y_h13d)
        tmp2 = tmp1 + 0.5 * self.dti * (v + P.Abs()(v)) / v * self.AZF(w) / dt * self.DZF(self.AZB(f)) / z_d[0] * \
               dz / self.AZB(dzz)
        tmp2 = P.Pad(((0, 0), (0, self.jm - 1), (1, 2)))(tmp2[:, 1:2, 1:self.kbm2])
        ff = ff * (1 - self.y_h3d * self.z_he3) + tmp2 * self.y_h3d * self.z_he3

        tmpff1 = ff[:, :1, :1]
        tmpff2 = ff[:, :1, self.kbm2:]
        tmpzero = P.Fill()(mstype.float32, (self.im, 1, self.kb - 3), 0)
        tmpff = P.Concat(2)((tmpff1, tmpzero, tmpff2))
        ff = P.Concat(1)((tmpff, ff[:, 1:, :]))
        tmpff1 = ff[:, -1:, :1]
        tmpff2 = ff[:, -1:, self.kbm2:]
        tmpff = P.Concat(2)((tmpff1, tmpzero, tmpff2))
        ff = P.Concat(1)((ff[:, :-1, :], tmpff))
        ff = ff * fsm

        fb = f + 0.5e0 * self.smoth * (ff + fb - 2e0 * f)
        f = ff

        return a, c, ee, gg, fb, f

    def construct(self, etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s,
                  rho, wubot, wvbot, ub, vb, egb, etb, dt, dhb, utb, vtb, vfluxb, et):
        """construct"""
        x_d, y_d, z_d = self.x_d, self.y_d, self.z_d
        q2b, q2lb = self.q2b, self.q2lb
        dx, dy = self.dx, self.dy

        # surface forcing
        w = w * (1 - self.z_h) + self.z_h * self.vfluxf
        # lateral_viscosity
        advx, advy, drhox, drhoy, aam = self.lateral_viscosity(dx, dy, u, v, dt, self.aam, ub, vb, x_d, y_d, z_d, rho,
                                                               self.rmean)
        # mode_interaction
        adx2d, ady2d, drx2d, dry2d, aam2d, advua, advva, egf, utf, vtf = self.mode_interaction(advx, advy, drhox, drhoy,
                                                                                               aam, x_d, y_d, d, uab,
                                                                                               vab, ua, va, el)

        # ===========external===========
        vamax = 0
        elf = 0
        for iext in range(1, 31):
            # external_el
            elf = self.external_el(x_d, y_d, d, ua, va, elb)
            # external_ua
            advua, uaf = self.external_ua(iext, x_d, y_d, elf, d, ua, va, uab, vab, el, elb, advua, aam2d, adx2d, drx2d,
                                          wubot)
            # external_va
            advva, vaf = self.external_va(iext, x_d, y_d, elf, d, ua, va, uab, vab, el, elb, advva, aam2d, ady2d, dry2d,
                                          wvbot)
            # external_update
            etf, uab, ua, vab, va, elb, el, d, egf, utf, vtf, vamax = self.external_update(iext, etf, ua, uab, va, vab,
                                                                                           el, elb, elf, uaf, vaf, egf,
                                                                                           utf, vtf)

        # ===========internal===========
        if self.global_step != 0:
            # adjust_uv
            u, v = self.adjust_uv(u, v, utb, vtb, utf, vtf, dt)
            # internal_w
            w = self.internal_w(x_d, y_d, dt, u, v, etf, etb, vfluxb)
            # internal_q
            dhf, a, c, gg, ee, kq, km, kh, q2b_, q2, q2lb_, q2l = self.internal_q(x_d, y_d, z_d, etf, aam, q2b, q2lb,
                                                                                  q2, q2l, kq, km, kh, u, v, w, dt, dhb,
                                                                                  rho, wubot, wvbot, t, s)
            q2b = P.Assign()(self.q2b, q2b_)
            q2lb = P.Assign()(self.q2lb, q2lb_)
            # internal_t_t
            a, c, ee, gg, tb, t = self.internal_t_(t, tb, self.wtsurf, self.tsurf, self.swrad, self.tclim, self.tbe,
                                                   self.tbw, self.tbn, self.tbs, x_d, y_d, z_d, dt, u, aam, self.h,
                                                   self.dum, v, self.dvm, w, dhf, etf, a, kh, self.dzz, c, self.dzz1,
                                                   ee, gg, dx, self.dz, dy, self.fsm, dhb)
            # internal_t_s
            a, c, ee, gg, sb, s = self.internal_t_(s, sb, self.wssurf, self.ssurf, self.swrad0, self.sclim, self.sbe,
                                                   self.sbw, self.sbn, self.sbs, x_d, y_d, z_d, dt, u, aam, self.h,
                                                   self.dum, v, self.dvm, w, dhf, etf, a, kh, self.dzz, c, self.dzz1,
                                                   ee, gg, dx, self.dz, dy, self.fsm, dhb)
            # dense
            rho = self.dens(s, t, self.zz, self.h, self.fsm)
            # internal_u
            uf, a, c, gg, ee, wubot = self.internal_u(x_d, z_d, dhf, u, v, w, ub, vb, egf, egb, ee, gg, self.cbc, km,
                                                      advx, drhox, dt, dhb)
            # internal_v
            vf, a, c, gg, ee, wvbot = self.internal_v(y_d, z_d, dhf, u, v, w, ub, vb, egf, egb, ee, gg, self.cbc, km,
                                                      advy, drhoy, dt, dhb)
            # adjust_ufvf
            u, v, ub, vb = self.adjust_ufvf(u, v, uf, vf, ub, vb)
        # internal_update
        egb, etb, dt, dhb, utb, vtb, vfluxb, et = self.internal_update(egf, etb, utf, vtf, etf, et)
        steps = P.AssignAdd()(self.global_step, 1)

        return elf, etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s, rho, wubot, wvbot, \
               ub, vb, egb, etb, dt, dhb, utb, vtb, vfluxb, et, steps, vamax, q2b, q2lb
