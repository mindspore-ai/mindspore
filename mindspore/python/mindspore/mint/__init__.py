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
"""mint module."""
from __future__ import absolute_import
from mindspore.ops.extend import *
from mindspore.ops.extend import array_func, math_func, nn_func
from mindspore.mint.nn.functional import *
from mindspore.mint.nn import functional
from mindspore.ops import erf, where
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
# 1

# 2

# 3

# 4

# 5

# 6

# 7

# 8

# 9

# 10

# 11

# 12

# 13

# 14

# 15

# 16

# 17

# 18

# 19

# 20
from mindspore.ops import prod
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

# 35

# 36

# 37

# 38

# 39

# 40

# 41

# 42

# 43

# 44

# 45

# 46

# 47

# 48

# 49

# 50

# 51

# 52

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

# 91

# 92

# 93

# 94

# 95

# 96

# 97

# 98

# 99

# 100

__all__ = [
    'full',
    'ones_like',
    'zeros_like',
    'erf',
    'where',
    'linspace',
    # 1

    # 2

    # 3

    # 4

    # 5

    # 6

    # 7

    # 8

    # 9

    # 10

    # 11

    # 12

    # 13

    # 14

    # 15

    # 16

    # 17

    # 18

    # 19

    # 20
    'prod',
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

    # 35

    # 36

    # 37

    # 38

    # 39

    # 40

    # 41

    # 42

    # 43

    # 44

    # 45

    # 46

    # 47

    # 48

    # 49

    # 50

    # 51

    # 52

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

    # 91

    # 92

    # 93

    # 94

    # 95

    # 96

    # 97

    # 98

    # 99

    # 100
]
__all__.extend(array_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
__all__.extend(functional.__all__)
