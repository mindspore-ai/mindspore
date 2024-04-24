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
A paralled predict example based on multiprocessing.
"""

from multiprocessing import Process, Pipe
import logging
from mslite_engine import MSLitePredict

ERROR = 0
PREDICT = 1
REPLY = 2
BUILD_FINISH = 3
EXIT = 4


def predict_process_func(pipe_child_end, model_path, device_id):
    """
    Function run in subprocess. Using pipe to communicate with main process.

    Args:
        pipe_child_end (multiprocessing.connection.Connection): A pipe to send or receive message.
        model_path (str): Mindir model path.
        device_id (int): Ascend device id to build and run model.
    """
    try:
        predict_worker = MSLitePredict(model_path, device_id)
        pipe_child_end.send((BUILD_FINISH, ""))
    except Exception as e:
        logging.exception(e)
        pipe_child_end.send((ERROR, str(e)))
        raise

    try:
        while True:
            msg_type, msg = pipe_child_end.recv()
            if msg_type == EXIT:
                print(f"Receive exit message {msg} and start to exit", flush=True)
                break
            if msg_type != PREDICT:
                raise RuntimeError(f"Expect to receive EXIT or PREDICT message for child process!")
            inputs = msg
            result = predict_worker.run(inputs)
            pipe_child_end.send((REPLY, result))

    except Exception as e:
        logging.exception(e)
        pipe_child_end.send((ERROR, str(e)))
        raise


class ParallelPredictUtils:
    """
    Class to control subprocess.
    """

    def __init__(self, model_path, device_num):
        self.model_path = model_path
        self.device_num = device_num
        self.pipe_parent_end = []
        self.pipe_child_end = []
        self.process = []
        for _ in range(device_num):
            pipe_end_0, pipe_end_1 = Pipe()
            self.pipe_parent_end.append(pipe_end_0)
            self.pipe_child_end.append(pipe_end_1)

    def _init_predict(self, device_id):
        process = Process(target=predict_process_func,
                          args=(self.pipe_child_end[device_id], self.model_path, device_id,))
        self.process.append(process)
        process.start()

    def build_model(self):
        for i in range(self.device_num):
            self._init_predict(i)
            msg_type, msg = self.pipe_parent_end[i].recv()
            if msg_type == ERROR:
                raise RuntimeError(f"Failed to build model {i}, exception occur: {msg}")
            assert msg_type == BUILD_FINISH
            print(f"Success to build model {i}")

    def run_predict(self, inputs):
        """
        Run predict in parallel.
        """
        result = []
        for i in range(self.device_num):
            self.pipe_parent_end[i].send((PREDICT, inputs[i]))

        for i in range(self.device_num):
            msg_type, msg = self.pipe_parent_end[i].recv()
            if msg_type == ERROR:
                raise RuntimeError(f"Failed to call predict, exception occur: {msg}")
            assert msg_type == REPLY
            result.append(msg)
        return result

    def finalize(self):
        for i in range(self.device_num):
            self.pipe_parent_end[i].send((EXIT, ""))
        for i in range(self.device_num):
            self.process[i].join()
