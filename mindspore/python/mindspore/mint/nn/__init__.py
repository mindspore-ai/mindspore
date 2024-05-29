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
"""
Neural Networks Cells.

Predefined building blocks or computing units to construct neural networks.
"""
from __future__ import absolute_import
from mindspore.nn.extend import *
from mindspore.nn.extend import basic, embedding
from mindspore.nn.extend import MaxPool2d
# 1

# 2

# 3

# 4

# 5

# 6
from mindspore.nn.layer.basic import UnfoldExt as Unfold
# 7
from mindspore.nn.layer.basic import Fold
# 8
from mindspore.nn.extend.layer import normalization
from mindspore.nn.extend.layer.normalization import *
# 9
from mindspore.nn.layer.basic import UpsampleExt as Upsample
# 10

# 11

# 12

# 13

# 14
from mindspore.nn.layer.basic import DropoutExt as Dropout
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
    'MaxPool2d',
    # 1

    # 2

    # 3

    # 4

    # 5

    # 6
    'Fold',
    # 7
    'Unfold',
    # 8

    # 9
    'Upsample',
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

__all__.extend(basic.__all__)
__all__.extend(embedding.__all__)
__all__.extend(normalization.__all__)
