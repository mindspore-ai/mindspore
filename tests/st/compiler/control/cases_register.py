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
import time
import operator


class Register:
    def __init__(self):
        self.case_targets = dict()
        self.case_levels = dict()
        self.skip_cases = dict()

    def target_ascend(self, fn):
        self._add_target(fn, "Ascend")
        return fn

    def target_gpu(self, fn):
        self._add_target(fn, "GPU")
        return fn

    def target_cpu(self, fn):
        self._add_target(fn, "CPU")
        return fn

    def level0(self, fn):
        self._add_level(fn, 0)
        return fn

    def level1(self, fn):
        self._add_level(fn, 1)
        return fn

    def skip(self, reason):
        def deco(fn):
            self.skip_cases[fn] = reason
            return fn

        return deco

    def _add_target(self, fn, target):
        if fn not in self.case_targets:
            self.case_targets[fn] = set()
        self.case_targets[fn].add(target)

    def _add_level(self, fn, level):
        self.case_levels[fn] = level

    def check_and_run(self, target, level):
        time_cost = dict()
        for fn, targets in self.case_targets.items():
            if fn in self.skip_cases:
                continue
            if target not in targets:
                continue
            if fn not in self.case_levels:
                continue
            if self.case_levels[fn] != level:
                continue
            print(f"\nexceute fn:{fn}, level:{level}, target:{target}")
            start_time = time.time()
            fn()
            end_time = time.time()
            time_cost[fn] = end_time - start_time

        sorted_time_cost = sorted(time_cost.items(), key=operator.itemgetter(1), reverse=True)
        total_cost_time = 0
        for item in sorted_time_cost:
            total_cost_time += item[1]
            print("Time:", item[1], ", fn:", item[0], "\n")
        print("Total cost time:", total_cost_time)


case_register = Register()
