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
bool Initialize(const std::string &backend, const std::string &global_group_name) {
  if (!InitializeCluster()) {
    MS_LOG(ERROR) << "Failed to initialize cluster.";
    return false;
  }

  if (!InitializeCollective(backend, global_group_name)) {
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

bool InitializeCluster() { return cluster::ClusterContext::instance()->Initialize(); }

bool FinalizeCluster() { return cluster::ClusterContext::instance()->Finalize(); }

bool InitializeCollective(const std::string &backend, const std::string &global_group_name) {
  return collective::CollectiveManager::instance()->Initialize(backend, global_group_name);
}

bool FinalizeCollective() { return collective::CollectiveManager::instance()->Finalize(); }
}  // namespace distributed
}  // namespace mindspore
