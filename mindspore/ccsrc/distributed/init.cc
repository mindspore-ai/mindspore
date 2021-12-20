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

#include "distributed/init.h"
#include <vector>
#include <string>

namespace mindspore {
namespace distributed {
bool Initialize() {
  if (!InitializeCluster()) {
    MS_LOG(ERROR) << "Failed to initialize cluster.";
    return false;
  }

#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (cluster::ClusterContext::instance()->initialized()) {
    // Server and Scheduler don't use collective communication library.
    auto node = cluster::ClusterContext::instance()->node();
    MS_EXCEPTION_IF_NULL(node);
    if (node->role() != ps::core::NodeRole::SCHEDULER) {
      // Global rank id and size should be manually set if cluster is initialized by MindSpore communication framework.
      auto abstract_node =
        std::dynamic_pointer_cast<ps::core::AbstractNode>(cluster::ClusterContext::instance()->node());
      MS_EXCEPTION_IF_NULL(abstract_node);
      collective::CollectiveManager::instance()->set_global_rank_id(abstract_node->rank_id());
      collective::CollectiveManager::instance()->set_global_rank_size(abstract_node->worker_num());

      if (!InitializeCollective()) {
        MS_LOG(ERROR) << "Failed to initialize collective communication.";
        return false;
      }
    }
  }
#endif
  return true;
}

bool Finalize() {
  if (!FinalizeCollective()) {
    MS_LOG(ERROR) << "Failed to finalize collective communication.";
    return false;
  }

  if (!FinalizeCluster()) {
    MS_LOG(ERROR) << "Failed to finalize cluster.";
    return false;
  }

  return true;
}

bool InitializeCluster() { return cluster::ClusterContext::instance()->Initialize(); }

bool FinalizeCluster() { return cluster::ClusterContext::instance()->Finalize(); }

bool InitializeCollective() { return collective::CollectiveManager::instance()->Initialize(); }

bool FinalizeCollective() { return collective::CollectiveManager::instance()->Finalize(); }
}  // namespace distributed
}  // namespace mindspore
