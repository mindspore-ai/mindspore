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

#include "plugin/cpu/res_manager/collective/allreduce_impl.h"

#include <vector>
#include <functional>
#include <memory>

namespace mindspore {
namespace device {
namespace cpu {
namespace {
constexpr size_t kWaitTimeout = 30;
}  // namespace

bool AllReduceLauncher::Initialize() {
  const auto &cluster_ctx = distributed::cluster::ClusterContext::instance();
  MS_EXCEPTION_IF_NULL(cluster_ctx);
  auto node_base = cluster_ctx->node_base();
  MS_EXCEPTION_IF_NULL(node_base);
  rank_id_ = node_base->rank_id();

  std::shared_ptr<distributed::cluster::topology::TcpNodeBase> client_node;
  client_node = std::dynamic_pointer_cast<distributed::cluster::topology::TcpNodeBase>(node_base);
  MS_EXCEPTION_IF_NULL(client_node);
  abs_node_ = std::make_shared<ps::core::CollectiveNode>(client_node);
  if (!abs_node_->Start()) {
    MS_LOG(ERROR) << "Failed to start the cpu collective node.";
    return false;
  }

  node_role_ = cluster_ctx->node_role();
  rank_size_ = static_cast<size_t>(cluster_ctx->node_num(cluster_ctx->node_role()));
  return true;
}

bool AllReduceLauncher::Finalize() {
  MS_EXCEPTION_IF_NULL(abs_node_);
  if (!abs_node_->Finish()) {
    MS_LOG(WARNING) << "Failed to finish the cpu collective node.";
  }
  if (!abs_node_->Stop()) {
    MS_LOG(ERROR) << "Failed to stop the cpu collective node.";
    return false;
  }
  return true;
}

const std::shared_ptr<ps::core::CollectiveNode> &AllReduceLauncher::collective_node() const { return abs_node_; }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
