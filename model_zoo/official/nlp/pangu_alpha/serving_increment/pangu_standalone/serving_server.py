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
# ============================================================================
"""Serving server start code, serve service, and manage all agents which load and execute models"""

import os
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(__file__))
    config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="pangu", device_ids=(0,))
    server.start_servables(config)

    server.start_grpc_server("127.0.0.1:5500")


if __name__ == "__main__":
    start()
