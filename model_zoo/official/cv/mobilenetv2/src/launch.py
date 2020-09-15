# Copyright 2020 Huawei Technologies Co., Ltd
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
"""launch train script"""
import os
import sys
import subprocess
import shutil
from args import launch_parse_args

def main():
    print("start", __file__)
    args = launch_parse_args()
    print(args)
    visible_devices = args.visible_devices.split(',')
    assert os.path.isfile(args.training_script)
    assert len(visible_devices) >= args.nproc_per_node
    print('visible_devices:{}'.format(visible_devices))

    # spawn the processes
    processes = []
    cmds = []
    log_files = []
    env = os.environ.copy()
    env['RANK_SIZE'] = str(args.nproc_per_node)
    cur_path = os.getcwd()
    for rank_id in range(0, args.nproc_per_node):
        os.chdir(cur_path)
        device_id = visible_devices[rank_id]
        rank_dir = os.path.join(cur_path, 'rank{}'.format(rank_id))
        env['RANK_ID'] = str(rank_id)
        env['DEVICE_ID'] = str(device_id)
        if os.path.exists(rank_dir):
            shutil.rmtree(rank_dir)
        os.mkdir(rank_dir)
        os.chdir(rank_dir)
        cmd = [sys.executable, '-u']
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)
        log_file = open(f'{rank_dir}/log{rank_id}.log', 'w')
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        processes.append(process)
        cmds.append(cmd)
        log_files.append(log_file)
    for process, cmd, log_file in zip(processes, cmds, log_files):
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process, cmd=cmd)
        log_file.close()


if __name__ == "__main__":
    main()
