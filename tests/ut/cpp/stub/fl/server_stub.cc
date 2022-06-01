/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fl/server/server.h"

namespace mindspore {
namespace fl {
namespace server {
Server &Server::GetInstance() {
  static Server instance;
  return instance;
}

void Server::Initialize(bool use_tcp, bool use_http, uint16_t http_port, const std::vector<RoundConfig> &rounds_config,
                        const CipherConfig &cipher_config, const FuncGraphPtr &func_graph, size_t executor_threshold) {}

void Server::Run() {}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
