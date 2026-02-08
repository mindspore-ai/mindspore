# Copyright 2025 Huawei Technologies Co., Ltd
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
Since it is necessary to check whether shared memory has been released,
the test cases cannot be run in parallel and thus are placed in the ST.
"""

import os
import subprocess
import time

import numpy as np
import psutil
from tests.mark_utils import arg_mark
import mindspore.dataset as ds


def check_message_queue(queue_input):
    cmd = f"ipcs -q -p | grep -E '\\b({queue_input})\\b' | wc -l"
    output = subprocess.check_output(cmd, shell=True).decode()
    lines = output.split('\n')
    assert lines[0] == str(0)


def check_shared_memory(memory_input):
    cmd = f"ipcs -m -p | grep -E '\\b({memory_input})\\b' | wc -l"
    output = subprocess.check_output(cmd, shell=True).decode()
    lines = output.split('\n')
    assert lines[0] == str(0)


def wait_independent_dataset_process_run():
    """waiting for independent_dataset_process run"""
    while True:
        current_process = psutil.Process()
        process_tree = current_process.children(recursive=True)
        # get the process name
        for process in process_tree:
            try:
                if "independent_dat" in process.name():
                    time.sleep(5)
                    return
            except psutil.NoSuchProcess:
                continue


def get_independent_dataset_process_pid():
    """Determine whether it includes a dataset independent process"""
    current_process = psutil.Process()
    process_tree = current_process.children(recursive=True)
    # get the process name
    for process in process_tree:
        if "independent_dat" in process.name():
            return process.pid
    raise RuntimeError("Couldn't find the independent dataset process.")


def check_pid(pid):
    """check the status of pid"""
    try:
        psutil.Process(pid)
        return True
    except psutil.NoSuchProcess:
        return False


def process_kill(pid, signum=9):
    """kill process"""
    if isinstance(pid, list):
        kill_cmd = "kill -{0} {1}".format(signum, " ".join(pid))
    elif isinstance(pid, int):
        kill_cmd = "kill -{0} {1}".format(signum, pid)
    else:
        raise TypeError("Type should be int or list (int)")
    subprocess.getstatusoutput(kill_cmd)

    time.sleep(2)
    pid_status = check_pid(pid)
    flag = not bool(pid_status)
    return flag


def check_processes_until(processes, timeout=60):
    """waiting for all the processes exit until timeout"""
    start = time.time()
    while (time.time() - start) <= timeout:
        all_exit_flag = True
        for item in processes:
            if check_pid(item):
                all_exit_flag = False
                break
        if all_exit_flag:
            return

    raise RuntimeError("All processes did not completely exit within " + str(timeout) + " seconds.")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dataset_two_stage_pipeline_with_big_data():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with big data in two stage pipeline
    Expectation: The dataset is processed as expected
    """
    os.environ['MS_INDEPENDENT_DATASET'] = 'True'
    ds.config.set_multiprocessing_start_method("spawn")

    independent_dataset_process_pid = None
    worker_processes_pids = None
    current_pid = os.getpid()

    class FirstDataset:
        """FirstDataset"""
        def __init__(self):
            self._image = np.ones((100, 2))
            self._label = np.zeros((100, 1))

        def __getitem__(self, index):
            video = np.random.random_sample((2, 720, 1280, 3))
            return video, self._label[index]

        @staticmethod
        def __len__():
            return 10

    first_dataset = ds.GeneratorDataset(source=FirstDataset(), column_names=["data", "label"],
                                        python_multiprocessing=True, num_parallel_workers=2)

    class SecondDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.dataset_size = self.dataset.get_dataset_size()
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

        def __next__(self):
            data = next(self.iterator)
            return data["data"], data["label"]

        def __iter__(self):
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
            return self

        def __len__(self):
            return self.dataset_size

    second_dataset = ds.GeneratorDataset(source=SecondDataset(first_dataset), column_names=["data", "label"],
                                         python_multiprocessing=True, num_parallel_workers=2)
    second_dataset = second_dataset.batch(5)
    count = 0
    for _ in second_dataset.create_dict_iterator(output_numpy=True):
        if count == 0:
            independent_dataset_process_pid = get_independent_dataset_process_pid()
            print("independent dataset process: {}".format(independent_dataset_process_pid), flush=True)

            worker_processes = psutil.Process(independent_dataset_process_pid).children(recursive=True)
            worker_processes_pids = [item.pid for item in worker_processes]
            print("worker processes: {}".format(worker_processes_pids), flush=True)

        count += 1

    check_processes_until([independent_dataset_process_pid])
    check_processes_until(worker_processes_pids)

    work_ids = [independent_dataset_process_pid] + worker_processes_pids + [current_pid]
    input_workers = "|".join(str(x) for x in work_ids)
    check_shared_memory(input_workers)
    check_message_queue(input_workers)

    ds.config.set_multiprocessing_start_method("fork")
    os.environ['MS_INDEPENDENT_DATASET'] = 'False'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_independent_generator_dataset_kill_main_process():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process when kill main process
    Expectation: The dataset is processed as expected
    """
    os.environ['MS_INDEPENDENT_DATASET'] = 'True'
    os.environ['GLOG_v'] = '1'

    log_file = "./test_dfx_independent_generator_dataset_kill_main_process.log"
    current_pid = os.getpid()

    # testcase process
    #        |-   /bin/sh -c python ./simple_dataset_pipeline.py True    ## os.popen()
    #                |-   main process
    #                          |-   independent dataset process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    ......

    kill_signal = [15, 9]
    parallel_mode = [False, True]
    for signal in kill_signal:
        for mode in parallel_mode:
            print(">>>> dataset in process mode: {}, kill main process with signal: {}".format(mode, signal),
                  flush=True)

            if os.path.exists(log_file):
                os.remove(log_file)

            cmd = "python ./simple_dataset_pipeline.py " + str(mode) + " >" + log_file + " 2>&1"
            os.popen(cmd)
            wait_independent_dataset_process_run()

            shell_process_pid = psutil.Process(current_pid).children(recursive=False)[0].pid
            main_process_pid = psutil.Process(shell_process_pid).children(recursive=False)[0].pid
            print("main process: {}".format(main_process_pid), flush=True)

            independent_dataset_process_pid = get_independent_dataset_process_pid()
            assert psutil.Process(main_process_pid).children(recursive=False)[0].pid == independent_dataset_process_pid
            print("independent dataset process: {}".format(independent_dataset_process_pid), flush=True)

            worker_processes = psutil.Process(independent_dataset_process_pid).children(recursive=True)
            worker_processes_pids = [item.pid for item in worker_processes]
            print("worker processes: {}".format(worker_processes_pids), flush=True)

            # kill -signal main process
            process_kill(main_process_pid, signal)

            # check the status of main process
            check_processes_until([main_process_pid])

            # if the independent dataset have exited
            check_processes_until([independent_dataset_process_pid])

            check_processes_until(worker_processes_pids)

            work_ids = [independent_dataset_process_pid] + worker_processes_pids + [main_process_pid]
            input_workers = "|".join(str(x) for x in work_ids)

            check_shared_memory(input_workers)
            check_message_queue(input_workers)

            # check the log
            ret = os.system(r"grep -RE '\[INFO\] .* Main process: " + str(main_process_pid) +
                            " had been closed' " + log_file)
            assert ret == 0

            if os.path.exists(log_file):
                os.remove(log_file)

    os.environ['MS_INDEPENDENT_DATASET'] = 'False'
    os.environ.pop('GLOG_v')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_independent_generator_dataset_kill_independent_dataset_process():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process when kill independent dataset process
    Expectation: The dataset is processed as expected
    """
    os.environ['MS_INDEPENDENT_DATASET'] = 'True'
    os.environ['GLOG_v'] = '1'

    log_file = "./test_dfx_independent_generator_dataset_kill_independent_dataset_process.log"
    current_pid = os.getpid()

    # testcase process
    #        |-   /bin/sh -c python ./simple_dataset_pipeline.py True    ## os.popen()
    #                |-   main process
    #                          |-   independent dataset process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    ......

    kill_signal = [15, 9]
    parallel_mode = [False, True]
    for signal in kill_signal:
        for mode in parallel_mode:
            print(">>>> dataset in process mode: {}, kill independent dataset process with signal: {}"
                  .format(mode, signal), flush=True)

            if os.path.exists(log_file):
                os.remove(log_file)

            cmd = "python ./simple_dataset_pipeline.py " + str(mode) + " >" + log_file + " 2>&1"
            os.popen(cmd)
            wait_independent_dataset_process_run()

            shell_process_pid = psutil.Process(current_pid).children(recursive=False)[0].pid
            main_process_pid = psutil.Process(shell_process_pid).children(recursive=False)[0].pid
            print("main process: {}".format(main_process_pid), flush=True)

            independent_dataset_process_pid = get_independent_dataset_process_pid()
            assert psutil.Process(main_process_pid).children(recursive=False)[0].pid == independent_dataset_process_pid
            print("independent dataset process: {}".format(independent_dataset_process_pid), flush=True)

            worker_processes = psutil.Process(independent_dataset_process_pid).children(recursive=True)
            worker_processes_pids = [item.pid for item in worker_processes]
            print("worker processes: {}".format(worker_processes_pids), flush=True)

            # kill -signal independent dataset process
            process_kill(independent_dataset_process_pid, signal)

            # check the status of main process
            check_processes_until([main_process_pid])

            # if the independent dataset have exited
            check_processes_until([independent_dataset_process_pid])

            check_processes_until(worker_processes_pids)

            work_ids = [independent_dataset_process_pid] + worker_processes_pids + [main_process_pid]
            input_workers = "|".join(str(x) for x in work_ids)

            check_shared_memory(input_workers)
            check_message_queue(input_workers)

            # check the log
            ret = os.system(r"grep -RE '\[ERROR\] \[Monitor\] The sub-process: " +
                            str(independent_dataset_process_pid) + " is terminated by a signal abnormally. Signal: " +
                            str(signal) + "' " + log_file)
            assert ret == 0

            if os.path.exists(log_file):
                os.remove(log_file)

    os.environ['MS_INDEPENDENT_DATASET'] = 'False'
    os.environ.pop('GLOG_v')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dfx_independent_generator_dataset_kill_worker_process():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process when kill worker process
    Expectation: The dataset is processed as expected
    """
    os.environ['MS_INDEPENDENT_DATASET'] = 'True'
    os.environ['GLOG_v'] = '1'

    log_file = "./test_dfx_independent_generator_dataset_kill_worker_process.log"
    current_pid = os.getpid()

    # testcase process
    #        |-   /bin/sh -c python ./simple_dataset_pipeline.py True    ## os.popen()
    #                |-   main process
    #                          |-   independent dataset process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    worker process
    #                                       |-    ......

    kill_signal = [15, 9]
    parallel_mode = [True]
    for signal in kill_signal:
        for mode in parallel_mode:
            print(">>>> dataset in process mode: {}, kill worker process with signal: {}".format(mode, signal),
                  flush=True)

            if os.path.exists(log_file):
                os.remove(log_file)

            cmd = "python ./simple_dataset_pipeline.py " + str(mode) + " >" + log_file + " 2>&1"
            os.popen(cmd)
            wait_independent_dataset_process_run()

            shell_process_pid = psutil.Process(current_pid).children(recursive=False)[0].pid
            main_process_pid = psutil.Process(shell_process_pid).children(recursive=False)[0].pid
            print("main process: {}".format(main_process_pid), flush=True)

            independent_dataset_process_pid = get_independent_dataset_process_pid()
            assert psutil.Process(main_process_pid).children(recursive=False)[0].pid == independent_dataset_process_pid
            print("independent dataset process: {}".format(independent_dataset_process_pid), flush=True)

            worker_processes = psutil.Process(independent_dataset_process_pid).children(recursive=True)
            worker_processes_pids = [item.pid for item in worker_processes]
            print("worker processes: {}".format(worker_processes_pids), flush=True)

            # kill -signal worker process
            process_kill(worker_processes_pids[3], signal)

            # check the status of main process
            check_processes_until([main_process_pid])

            # if the independent dataset have exited
            check_processes_until([independent_dataset_process_pid])

            check_processes_until(worker_processes_pids)

            work_ids = [independent_dataset_process_pid] + worker_processes_pids + [main_process_pid]
            input_workers = "|".join(str(x) for x in work_ids)

            check_shared_memory(input_workers)
            check_message_queue(input_workers)

            # check the log
            signal_str = "Terminated"
            if signal == 9:
                signal_str = "Killed"
            ret = os.system(r"grep -RE '\[ERROR\] .* Dataset worker process " + str(worker_processes_pids[3]) +
                            " was killed by signal: " + signal_str + "' " + log_file)
            assert ret == 0

            if os.path.exists(log_file):
                os.remove(log_file)

    os.environ['MS_INDEPENDENT_DATASET'] = 'False'
    os.environ.pop('GLOG_v')


if __name__ == "__main__":
    test_dataset_two_stage_pipeline_with_big_data()
    test_dfx_independent_generator_dataset_kill_main_process()
    test_dfx_independent_generator_dataset_kill_independent_dataset_process()
    test_dfx_independent_generator_dataset_kill_worker_process()
