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

#include "runtime/graph_scheduler/actor/rpc/rpc_actor.h"

namespace mindspore {
namespace runtime {
void RpcActor::SetOpcontext(OpContext<DeviceTensor> *const op_context) { op_context_ = op_context; }

void RpcActor::set_actor_route_table_proxy(const ActorRouteTableProxyPtr &proxy) { actor_route_table_proxy_ = proxy; }

void RpcActor::set_inter_process_edge_name(const std::string &edge_name) { inter_process_edge_name_ = edge_name; }
}  // namespace runtime
}  // namespace mindspore
