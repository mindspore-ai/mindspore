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
#include "include/backend/distributed/cluster/topology/node_base.h"
#include "include/backend/distributed/cluster/cluster_context.h"

namespace mindspore {
namespace distributed {
namespace cluster {
ClusterContext::ClusterContext()
    : inited_(false),
      finalized_(true),
      node_num_each_role_({}),
      scheduler_host_(kLocalHost),
      scheduler_port_(kDefaultSchedPort),
      node_role_(""),
      cluster_config_(nullptr) {}

ClusterContext::~ClusterContext() {
  if (!finalized_) {
    try {
      (void)Finalize();
    } catch (std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize cluster context.";
    }
  }
  finalized_ = true;
}
bool ClusterContext::Initialize() { return true; }
bool ClusterContext::Finalize(uint32_t timeout) { return true; }
const std::shared_ptr<topology::NodeBase> &ClusterContext::node() const { return node_base_; }
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
