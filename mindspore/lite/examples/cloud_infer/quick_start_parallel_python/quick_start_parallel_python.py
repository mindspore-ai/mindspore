# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""quick_start_cloud_infer_parallel_python."""

import time
from threading import Thread
import numpy as np
import mindspore_lite as mslite

# Use case: serving inference.
# Precondition 1: Download MindSpore Lite serving package or building MindSpore Lite serving package by
#                 export MSLITE_ENABLE_SERVER_INFERENCE=on.
# Precondition 2: Install wheel package of MindSpore Lite built by precondition 1.
# The result can be find in the tutorial of runtime_parallel_python.
# the number of threads of one worker.
# WORKERS_NUM * THREAD_NUM should not exceed the number of cores of the machine.
THREAD_NUM = 1
# In parallel inference, the number of workers in one `ModelParallelRunner` in server.
# If you prepare to compare the time difference between parallel inference and serial inference,
# you can set WORKERS_NUM = 1 as serial inference.
WORKERS_NUM = 3
# Simulate 5 clients, and each client sends 2 inference tasks to the server at the same time.
PARALLEL_NUM = 5
TASK_NUM = 2


def parallel_runner_predict(parallel_runner, parallel_id):
    """
    One Runner with 3 workers, set model input, execute inference and get output.

    Args:
        parallel_runner (mindspore_lite.ModelParallelRunner): Actuator Supporting Parallel inference.
        parallel_id (int): Simulate which client's task to process
    """

    task_index = 0
    while True:
        if task_index == TASK_NUM:
            break
        task_index += 1
        # Set model input
        inputs = parallel_runner.get_inputs()
        in_data = np.fromfile("./model/input.bin", dtype=np.float32)
        inputs[0].set_data_from_numpy(in_data)
        once_start_time = time.time()
        # Execute inference
        outputs = parallel_runner.predict(inputs)
        once_end_time = time.time()
        print("parallel id: ", parallel_id, " | task index: ", task_index, " | run once time: ",
              once_end_time - once_start_time, " s")
        # Get output
        for output in outputs:
            tensor_name = output.name.rstrip()
            data_size = output.data_size
            element_num = output.element_num
            print("tensor name is:%s tensor size is:%s tensor elements num is:%s" % (tensor_name,
                                                                                     data_size,
                                                                                     element_num))
            data = output.get_data_to_numpy()
            data = data.flatten()
            print("output data is:", end=" ")
            for j in range(5):
                print(data[j], end=" ")
            print("")


# Init RunnerConfig and context, and add CPU device info
context = mslite.Context()
context.target = ["cpu"]
context.cpu.thread_num = THREAD_NUM
context.cpu.inter_op_parallel_num = THREAD_NUM
context.parallel.workers_num = WORKERS_NUM
# Build ModelParallelRunner from file
model_parallel_runner = mslite.ModelParallelRunner()
model_parallel_runner.build_from_file(model_path="./model/mobilenetv2.mindir", context=context)
# The server creates 5 threads to store the inference tasks of 5 clients.
threads = []
total_start_time = time.time()
for i in range(PARALLEL_NUM):
    threads.append(Thread(target=parallel_runner_predict, args=(model_parallel_runner, i,)))
# Start threads to perform parallel inference.
for th in threads:
    th.start()
for th in threads:
    th.join()
total_end_time = time.time()
print("total run time: ", total_end_time - total_start_time, " s")
