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

""" define marks """
import pytest


def arg_mark(plat_marks, level_mark, card_mark, essential_mark):
    optional_plat_marks = ['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                           'cpu_macos']
    optional_level_marks = ['level0', 'level1', 'level2', 'level3', 'level4']
    optional_card_marks = ['onecard', 'allcards']
    optional_essential_marks = ['essential', 'unessential']
    if not plat_marks or not set(plat_marks).issubset(set(optional_plat_marks)):
        raise ValueError("wrong plat_marks values")
    if level_mark not in optional_level_marks:
        raise ValueError("wrong level_mark value")
    if card_mark not in optional_card_marks:
        raise ValueError("wrong card_mark value")
    if essential_mark not in optional_essential_marks:
        raise ValueError("wrong essential_mark value")

    def decorator(func):
        for plat_mark in plat_marks:
            func = getattr(pytest.mark, plat_mark)(func)
        func = getattr(pytest.mark, level_mark)(func)
        func = getattr(pytest.mark, card_mark)(func)
        func = getattr(pytest.mark, essential_mark)(func)
        return func

    return decorator
