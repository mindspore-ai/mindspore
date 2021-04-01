# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Watchpoints test script for offline debugger APIs.
"""

import mindspore.offline_debug.dbg_services as d


def main():

    debugger_backend = d.DbgServices(
        dump_file_path="/opt/nvme2n1/j00455527/dumps/async_sink_true/032421")

    _ = debugger_backend.initialize(net_name="alexnet", is_sync_mode=False)

    # NOTES:
    # -> watch_condition=6 is MIN_LT
    # -> watch_condition=18 is CHANGE_TOO_LARGE

    # test 1: watchpoint set and hit (watch_condition=6)
    param1 = d.Parameter(name="param", disabled=False, value=0.0)
    _ = debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=6,
                                        check_node_list={"Default/network-TrainOneStepCell/network-WithLossCell/"
                                                         "_backbone-AlexNet/conv3-Conv2d/Conv2D-op169":
                                                         {"device_id": [0], "root_graph_id": [1], "is_parameter": False
                                                          }}, parameter_list=[param1])

    watchpoint_hits_test_1 = debugger_backend.check_watchpoints(iteration=2)
    if len(watchpoint_hits_test_1) != 1:
        print("ERROR -> test 1: watchpoint set but not hit just once")
    print_watchpoint_hits(watchpoint_hits_test_1, 1)

    # test 2: watchpoint remove and ensure it's not hit
    _ = debugger_backend.remove_watchpoint(watchpoint_id=1)
    watchpoint_hits_test_2 = debugger_backend.check_watchpoints(iteration=2)
    if watchpoint_hits_test_2:
        print("ERROR -> test 2: watchpoint removed but hit")

    # test 3: watchpoint set and not hit, then remove
    param2 = d.Parameter(name="param", disabled=False, value=-1000.0)
    _ = debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                        check_node_list={"Default/network-TrainOneStepCell/network-WithLossCell/"
                                                         "_backbone-AlexNet/conv3-Conv2d/Conv2D-op169":
                                                         {"device_id": [0], "root_graph_id": [1], "is_parameter": False
                                                          }}, parameter_list=[param2])

    watchpoint_hits_test_3 = debugger_backend.check_watchpoints(iteration=2)
    if watchpoint_hits_test_3:
        print("ERROR -> test 3: watchpoint set but not supposed to be hit")
    _ = debugger_backend.remove_watchpoint(watchpoint_id=2)


def print_watchpoint_hits(watchpoint_hits, test_id):
    """Print watchpoint hits."""
    for x, _ in enumerate(watchpoint_hits):
        print("-----------------------------------------------------------")
        print("watchpoint_hit for test_%u attributes:" % test_id)
        print("name = ", watchpoint_hits[x].name)
        print("slot = ", watchpoint_hits[x].slot)
        print("condition = ", watchpoint_hits[x].condition)
        print("watchpoint_id = ", watchpoint_hits[x].watchpoint_id)
        for p, _ in enumerate(watchpoint_hits[x].parameters):
            print("parameter ", p, " name = ",
                  watchpoint_hits[x].parameters[p].name)
            print("parameter ", p, " disabled = ",
                  watchpoint_hits[x].parameters[p].disabled)
            print("parameter ", p, " value = ",
                  watchpoint_hits[x].parameters[p].value)
            print("parameter ", p, " hit = ",
                  watchpoint_hits[x].parameters[p].hit)
            print("parameter ", p, " actual_value = ",
                  watchpoint_hits[x].parameters[p].actual_value)
        print("error code = ", watchpoint_hits[x].error_code)
        print("device_id = ", watchpoint_hits[x].device_id)
        print("root_graph_id = ", watchpoint_hits[x].root_graph_id)


if __name__ == "__main__":
    main()
