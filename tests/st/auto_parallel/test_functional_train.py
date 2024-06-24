# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level2", card_mark="allcards",
          essential_mark="essential")
def test_pynative_functional_train():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: pynative mode
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v functional_train.py::test_pynative_func")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_graph_functional_train():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: graph mode
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v functional_train.py::test_graph_func")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_graph_functional_sink_train():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: graph mode, data sink
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v functional_train.py::test_graph_func_sink")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_pynative_func_sink():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: pynative mode, data sink with jit
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v functional_train.py::test_pynative_func_sink")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_shard_func():
    '''
    Feature: shard func in pynative mode
    Description: pynative mode, shard func
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v functional_train.py::test_shard_func")
    assert ret == 0
