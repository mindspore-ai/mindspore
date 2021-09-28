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
"""kernel build server for ascend"""
import sys
import warnings
import json

from mindspore._extends.parallel_compile.tbe_compiler.tbe_job_manager import TbeJobManager
from mindspore._extends.remote.kernel_build_server import Messager, get_logger, AkgBuilder


class AscendMessager(Messager):
    """
    Ascend Messager
    It works as a server, communicating with c++ client.
    """

    def __init__(self, fdin, fdout):
        super().__init__(fdin, fdout)
        get_logger().info("[TRACE] Ascend Messager init...")
        self.tbe_builder = TbeJobManager()
        self.akg_builder = AkgBuilder("ASCEND")

    def handle(self):
        """
        Communicate with remote client.
        Reference protocol between them at PR#3821 and PR#3935
        """
        arg = self.get_message()
        if arg.startswith('AKG'):
            self.akg_builder.handle(self, arg)
        else:
            job_json = dict()
            try:
                job_json = json.loads(arg)
            except json.decoder.JSONDecodeError:
                get_logger().error("[TRACE] Request is not a json message: {}".format(arg))
                self.send_ack(False)
                self.exit()
            finally:
                pass

            if "job_type" in job_json:
                res = self.tbe_builder.job_handler(arg)
                self.send_res(res)
            else:
                get_logger().error("[TRACE] Request is not a TBE Job message: {}".format(arg))
                self.send_ack(False)
                self.exit()

    def exit(self):
        self.tbe_builder.reset()
        get_logger().info("[TRACE] Ascend Messager Exit...")
        exit()


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    if len(sys.argv) != 3:
        raise Exception('Incorrect argv: {}'.format(sys.argv))
    get_logger().debug(f"[TRACE] argv: {str(sys.argv)}")
    messager = AscendMessager(int(sys.argv[1]), int(sys.argv[2]))
    messager.run()
