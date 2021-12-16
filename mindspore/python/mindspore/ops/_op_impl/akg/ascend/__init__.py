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

"""__init__"""

from .abs import _abs_akg
from .add import _add_akg
from .add_n import _addn_akg
from .batchmatmul import _batchmatmul_akg
from .cast import _cast_akg
from .equal import _equal_akg
from .exp import _exp_akg
from .expand_dims import _expand_dims_akg
from .greater import _greater_akg
from .greater_equal import _greater_equal_akg
from .less import _less_akg
from .less_equal import _less_equal_akg
from .log import _log_akg
from .maximum import _maximum_akg
from .minimum import _minimum_akg
from .mul import _mul_akg
from .neg import _neg_akg
from .pow import _power_akg
from .real_div import _real_div_akg
from .reciprocal import _reciprocal_akg
from .reduce_max import _reduce_max_akg
from .reduce_min import _reduce_min_akg
from .reduce_sum import _reduce_sum_akg
from .rsqrt import _rsqrt_akg
from .select import _select_akg
from .sqrt import _sqrt_akg
from .square import _square_akg
from .sub import _sub_akg
from .prod_force_se_a import _prod_force_se_a_akg
from .load_im2col import _load_im2col_akg

# Please insert op register in lexicographical order of the filename.
