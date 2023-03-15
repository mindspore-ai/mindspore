/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <string>
#include "include/backend/distributed/cluster/dummy_cluster_context.h"

namespace mindspore {
namespace distributed {
namespace cluster {
std::shared_ptr<ClusterContext> ClusterContext::instance() {
  static std::shared_ptr<ClusterContext> cluster_instance = nullptr;
  if (cluster_instance == nullptr) {
    cluster_instance.reset(new (std::nothrow) ClusterContext());
    MS_EXCEPTION_IF_NULL(cluster_instance);
  }
  return cluster_instance;
}

bool ClusterContext::Initialize() const { return true; }

bool ClusterContext::Finalize(uint32_t) const { return true; }

std::string ClusterContext::node_role() const { return ""; }

uint32_t ClusterContext::node_num(const std::string &) { return 0; }

bool ClusterContext::initialized() const { return false; }

void ClusterContext::set_cluster_exit_with_exception() { return; }

bool ClusterContext::cluster_exit_with_exception() const { return true; }
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
