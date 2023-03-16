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

#ifndef MINDSPORE_CCSRC_PS_CORE_RECOVERY_BASE_H_
#define MINDSPORE_CCSRC_PS_CORE_RECOVERY_BASE_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "include/backend/distributed/ps/constants.h"
#include "utils/log_adapter.h"
#include "ps/core/file_configuration.h"
#include "include/backend/distributed/ps/ps_context.h"

namespace mindspore {
namespace ps {
namespace core {
enum class StorageType : int { kFileStorage = 1 };
// RecoveryBase is used to parse configuration items related to recovery.
// It is the base class of SchedulerRecovery and NodeRecovery.
class RecoveryBase {
 public:
  RecoveryBase() : recovery_storage_(nullptr), storage_type_(StorageType::kFileStorage) {}

  virtual ~RecoveryBase() = default;

  // Initialize the recovery configuration item and get the storage type of recovery.
  virtual bool Initialize(const std::string &json_config);

  // Initialize the recovery configuration item and get the storage type of recovery.
  virtual bool InitializeNodes(const std::string &json_config);

  // The node needs to recover metadata information when it starts.
  virtual bool Recover() = 0;

  // Persist metadata to storage.
  virtual void Persist(const core::ClusterConfig &clusterConfig);

  // Persist metadata to storage.
  virtual void PersistNodesInfo(const core::ClusterConfig &clusterConfig);

 protected:
  // Persistent storage used to save metadata.
  std::unique_ptr<Configuration> recovery_storage_;

  // Persistent storage used to save server nodes metadata.
  std::unique_ptr<Configuration> scheduler_recovery_storage_;

  // Storage type for recovery,Currently only supports storage of file types
  StorageType storage_type_;

  std::mutex recovery_mtx_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_RECOVERY_BASE_H_
