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

#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"

namespace mindspore {
namespace runtime {
void RecvActor::SetRouteInfo(uint32_t, const std::string &, const std::string &src_node_name, const std::string &) {
  input_peer_node_name_.emplace_back(src_node_name);
}
}  // namespace runtime
}  // namespace mindspore
