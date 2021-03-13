# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""kernel build server for gpu"""
import os
import sys
import warnings
from mindspore._extends.remote.kernel_build_server import Messager, get_logger, AkgBuilder
from mindspore._extends.parallel_compile.akg_compiler.compiler import run_compiler as akg_compile_single

class GpuMessager(Messager):
    '''
    GPU Messager
    It works as a server, communicating with c++ client.
    '''

    def __init__(self, fdin, fdout):
        super().__init__(fdin, fdout)
        get_logger().info("[TRACE] GPU Messager init...")
        self.akg_builder = AkgBuilder()

    def handle(self):
        """
        Communicate with remote client.
        Reference protocol between them at PR#4063
        """
        arg = self.get_message()
        if arg == 'AKG/PID':
            self.send_res(os.getpid())
        elif arg == 'AKG/START':
            self.send_ack()
            process_num_str = self.get_message()
            self.send_ack()
            wait_time_str = self.get_message()
            self.akg_builder.create(int(process_num_str), int(wait_time_str), "GPU")
            self.send_ack()
        elif arg == 'AKG/DATA':
            self.send_ack()
            while True:
                req = self.get_message()
                if req.startswith('{'):
                    self.akg_builder.accept_json(req)
                    self.send_ack()
                elif req == 'AKG/WAIT':
                    res = self.akg_builder.compile()
                    self.send_res(res)
                    break
                else:
                    self.send_ack(False)
                    break
        elif arg == 'AKG/COMPILE':
            self.send_ack()
            json = self.get_message()
            try:
                akg_compile_single(json)
            except ValueError:
                self.send_ack(False)
                self.exit()
            finally:
                pass
            self.send_ack()
        else:
            self.send_ack(False)
            self.exit()

    def exit(self):
        get_logger().info("[TRACE] GPU Messager Exit...")
        exit()

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    if len(sys.argv) != 3:
        raise Exception('Incorrect argv: {}'.format(sys.argv))
    get_logger().debug(f"[TRACE] argv: {str(sys.argv)}")
    messager = GpuMessager(int(sys.argv[1]), int(sys.argv[2]))
    messager.run()
