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
#include "distributed/recovery/recovery_context.h"

namespace mindspore {
namespace distributed {
using distributed::recovery::RecoveryContext;

bool Initialize() {
  // If this process participates in the cluster building, we need to initialize cluster context.
  if (common::UseDynamicCluster()) {
    if (!InitializeCluster()) {
      MS_LOG(ERROR) << "Failed to initialize distributed training.";
      return false;
    }
  }

  // Initialize the collective manager regardless of whether the cluster is initialized or not.
  if (!InitializeCollective()) {
    MS_LOG(ERROR) << "Failed to initialize collective communication.";
    return false;
  }
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

bool InitializeCluster() {
  if (!cluster::ClusterContext::instance()->Initialize()) {
    MS_LOG(ERROR) << "Failed to initialize cluster.";
    return false;
  }
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  auto node = cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);

  // Set the callback for the cluster node.
  auto callback = std::make_shared<std::function<void(void)>>([]() {
    if (!collective::CollectiveManager::instance()->Finalize()) {
      MS_LOG(EXCEPTION) << "Failed to finalize the collective communication lib.";
    }
  });
  node->set_abnormal_callback(callback);

  if (cluster::ClusterContext::instance()->initialized() && !collective::CollectiveManager::instance()->initialized()) {
    // Scheduler don't use collective communication library.
    const auto &cluster_ctx = cluster::ClusterContext::instance();
    MS_EXCEPTION_IF_NULL(cluster_ctx);
    if (cluster_ctx->node_role() != kEnvRoleOfScheduler) {
      // Global rank id and size should be manually set if cluster is initialized by MindSpore communication framework.
      collective::CollectiveManager::instance()->set_global_rank_id(node->rank_id());
      auto global_rank_size = cluster_ctx->node_num(cluster_ctx->node_role());
      collective::CollectiveManager::instance()->set_global_rank_size(global_rank_size);

      if (RecoveryContext::GetInstance()->enable_recovery()) {
        RecoveryContext::GetInstance()->set_global_rank_id(node->rank_id());
        RecoveryContext::GetInstance()->set_global_rank_size(global_rank_size);
      }

      if (RecoveryContext::GetInstance()->enable_recovery()) {
        RecoveryContext::GetInstance()->ObtainGlobalLatestCkptInfo();
      }
    }
  }
#endif
  return true;
}

bool FinalizeCluster() { return cluster::ClusterContext::instance()->Finalize(); }

bool InitializeCollective() {
  if (collective::CollectiveManager::instance()->initialized()) {
    return true;
  }
  if (cluster::ClusterContext::instance()->initialized() &&
      cluster::ClusterContext::instance()->node_role() == kEnvRoleOfScheduler) {
    MS_LOG(INFO) << "Scheduler node does not need to initialize collective communication.";
    return true;
  }
  return collective::CollectiveManager::instance()->Initialize();
}

bool FinalizeCollective() { return collective::CollectiveManager::instance()->Finalize(); }
}  // namespace distributed
}  // namespace mindspore
