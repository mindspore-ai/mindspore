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
"""read variables"""

import numpy as np
import netCDF4 as nc


# variable name list
params_name = ['z', 'zz', 'dz', 'dzz', 'dx', 'dy', 'cor', 'h', 'fsm', 'dum', 'dvm', 'art', 'aru', 'arv', 'rfe', 'rfw',
               'rfn', 'rfs', 'east_e', 'north_e', 'east_c', 'north_c', 'east_u', 'north_u', 'east_v', 'north_v', 'tb',
               'sb', 'tclim', 'sclim', 'rot', 'vfluxf', 'wusurf', 'wvsurf', 'e_atmos', 'ub', 'vb', 'uab', 'vab', 'elb',
               'etb', 'dt', 'uabw', 'uabe', 'vabs', 'vabn', 'els', 'eln', 'ele', 'elw', 'ssurf', 'tsurf', 'tbe', 'sbe',
               'sbw', 'tbw', 'tbn', 'tbs', 'sbn', 'sbs', 'wtsurf', 'swrad']


def load_var(file_obj, name):
    """load variable from nc data file"""
    data = file_obj.variables[name]
    data = data[:]
    data = np.float32(np.transpose(data, (2, 1, 0)))
    return data


def read_nc(file_path):
    """ put the load variable into the dict """
    variable = {}
    file_obj = nc.Dataset(file_path)
    for name in params_name:
        variable[name] = load_var(file_obj, name)
    return variable
