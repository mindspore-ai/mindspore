# Copyright 2024 Huawei Technologies Co., Ltd
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
"""mint nn functional."""
from __future__ import absolute_import
from mindspore.ops.extend import max_pool2d
from mindspore.ops.functional import (
    conv_transpose2d,
    grid_sample
)
# 1

# 2

# 3

# 4

# 5
from mindspore.ops.functional import pad_ext as pad
# 6

# 7

# 8
from mindspore.ops.functional import layer_norm
# 9
from mindspore.ops.function.nn_func import interpolate_ext as interpolate
# 10

# 11
from mindspore.ops.functional import relu
# 12

# 13

# 14
from mindspore.ops.function.nn_func import dropout_ext as dropout
# 15

# 16

# 17

# 18

# 19

# 20

# 21

# 22

# 23

# 24

# 25

# 26

# 27

# 28

# 29

# 30

# 31

# 32

# 33

# 34
from mindspore.ops.function.nn_func import batch_norm_ext as batch_norm
# 35

# 36
from mindspore.ops.functional import gelu
# 37

# 38

# 39
from mindspore.ops.functional import group_norm
# 40

# 41

# 42

# 43

# 44

# 45

# 46
from mindspore.ops.functional import silu
# 47

# 48

# 49
from mindspore.ops.functional import sigmoid
# 50

# 51

# 52
from mindspore.ops.functional import embedding
# 53

# 54

# 55

# 56

# 57

# 58

# 59

# 60

# 61

# 62

# 63

# 64

# 65

# 66

# 67

# 68

# 69

# 70

# 71

# 72

# 73

# 74

# 75

# 76

# 77

# 78

# 79

# 80

# 81

# 82

# 83

# 84

# 85

# 86

# 87

# 88

# 89

# 90
from mindspore.ops.function.nn_func import avg_pool2d_ext as avg_pool2d
# 91

# 92
from mindspore.ops.extend import leaky_relu_ext as leaky_relu
# 93
from mindspore.ops.function.nn_func import softplus_ext as softplus
# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99

# 100

__all__ = [
    'conv_transpose2d',
    'max_pool2d',
    # 1

    # 2

    # 3

    # 4

    # 5
    'pad',
    # 6

    # 7

    # 8
    'layer_norm',
    # 9
    'interpolate',
    # 10

    # 11
    'relu',
    # 12

    # 13

    # 14
    'dropout',
    # 15

    # 16

    # 17

    # 18

    # 19

    # 20

    # 21

    # 22

    # 23

    # 24

    # 25

    # 26

    # 27

    # 28

    # 29

    # 30

    # 31

    # 32

    # 33

    # 34
    'batch_norm',
    # 35

    # 36
    'gelu',
    # 37

    # 38

    # 39
    'group_norm',
    # 40

    # 41

    # 42

    # 43

    # 44

    # 45

    # 46
    'silu',
    # 47

    # 48

    # 49
    'sigmoid',
    # 50

    # 51

    # 52
    'embedding',
    # 53

    # 54

    # 55

    # 56

    # 57

    # 58

    # 59

    # 60

    # 61

    # 62

    # 63

    # 64

    # 65

    # 66

    # 67

    # 68

    # 69

    # 70

    # 71

    # 72

    # 73

    # 74

    # 75

    # 76

    # 77

    # 78

    # 79

    # 80

    # 81

    # 82

    # 83

    # 84

    # 85

    # 86

    # 87

    # 88

    # 89

    # 90
    'avg_pool2d',
    # 91
    'grid_sample',
    # 92
    'leaky_relu',
    # 93
    'softplus',
    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100
]
