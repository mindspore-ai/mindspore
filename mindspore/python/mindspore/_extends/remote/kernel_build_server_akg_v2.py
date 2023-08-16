# Copyright 2023 Huawei Technologies Co., Ltd
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
"""kernel build server for akg v2 kernels"""
import sys
import warnings
from mindspore._extends.remote.kernel_build_server import Messager, get_logger, AkgV2Builder


class AkgV2Messager(Messager):
    '''
    Default Messager for akg v2 kernels.
    It works as a server, communicating with c++ client.
    '''

    def __init__(self, fdin, fdout):
        super().__init__(fdin, fdout)
        get_logger().info("[TRACE] AKG V2 Messager init...")
        self.akg_v2_builder = AkgV2Builder("default")

    def handle(self):
        """
        Communicate with remote client.
        Reference protocol between them at PR#4063
        """
        arg = self.get_message()
        if "AKG" in arg:
            self.akg_v2_builder.handle(self, arg)
        else:
            self.send_ack(False)
            self.exit()

    def exit(self):
        get_logger().info("[TRACE] AKG V2 Messager Exit...")
        exit()


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    if len(sys.argv) != 3:
        raise Exception(f"Incorrect argv: {sys.argv}")
    get_logger().debug(f"[TRACE] argv: {str(sys.argv)}")
    messager = AkgV2Messager(int(sys.argv[1]), int(sys.argv[2]))
    messager.run()
