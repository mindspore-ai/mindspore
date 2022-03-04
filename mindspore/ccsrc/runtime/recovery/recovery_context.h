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

#ifndef MINDSPORE_CCSRC_RUNTIME_RECOVERY_RECOVERY_H_
#define MINDSPORE_CCSRC_RUNTIME_RECOVERY_RECOVERY_H_

#include <vector>
#include <string>
#include <memory>
#include "utils/ms_utils.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/persistent/storage/json_utils.h"
#include "runtime/collective/collective_communication_lib.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
namespace recovery {
using distributed::storage::FileIOUtils;
using distributed::storage::JsonUtils;

enum class RecoveryErrCode { kUnKnownError, kAllGatherHostNameFailed, kBroadcastUniqueIDFailed, kInitNcclFailed };

// Used to save disaster recovery-related state quantities and provide disaster recovery-related
// functions, such as reinitializing collective communication, etc.
class BACKEND_EXPORT RecoveryContext {
 public:
  static std::shared_ptr<RecoveryContext> &GetInstance() {
    if (instance_ == nullptr) {
      instance_.reset(new (std::nothrow) RecoveryContext());
      instance_->Initialize();
    }
    return instance_;
  }
  ~RecoveryContext() = default;

  // Reinitializing collective communication.
  bool ReInitializeCollective();

  // Obtain the global step corresponding to the global latest checkpoint in each training process. Since there may be
  // some processes that fails to save the checkpoint, it is necessary for AllGather to save the latest step of the
  // successful checkpoint in each training process, and then take the minimum value as the final reset position.
  void ObtainGlobalLatestCkptInfo();

  // Get whether enable recovery or not.
  bool enable_recovery() const { return enable_recovery_; }

  // Get the persistent directory.
  const std::string &recovery_path() const { return recovery_path_; }

  // Get interval to persist model.
  int recovery_interval() const { return recovery_interval_; }

  // Get the error status of recovery.
  RecoveryErrCode recovery_status() const { return recovery_status_; }
  // Set the error status of recovery.
  void set_recovery_status(RecoveryErrCode recovery_status) { recovery_status_ = recovery_status; }

  // Set the path used to save checkpoint.
  void SetCkptPath(const std::string &path);
  // Get the path used to save checkpoint.
  std::string GetCkptPath();

  // Get the latest checkpoint in this node.
  std::string latest_ckpt_file() const { return latest_ckpt_file_; }
  // Get the epoch of latest checkpoint in this node.
  int latest_ckpt_epoch() const { return latest_ckpt_epoch_; }
  // Get the step of latest checkpoint in this node.
  int latest_ckpt_step() const { return latest_ckpt_step_; }

  // Set whether need to reset training process or not, if true, all training process need to rollback the same step of
  // latest checkpoint, including loading checkpoint and reset the minddata.
  void set_need_reset(bool need_reset) { need_reset_ = need_reset; }
  // Get whether need to reset training process or not.
  bool need_reset() const { return need_reset_; }

  // Set whether need to sync the weight of model to device.
  void set_need_sync_weight_to_device(bool need_sync_weight_to_device) {
    need_sync_weight_to_device_ = need_sync_weight_to_device;
  }
  // Get whether need to sync the weight of model to device or not.
  bool need_sync_weight_to_device() const { return need_sync_weight_to_device_; }

  // Set global rank id.
  void set_global_rank_id(uint32_t global_rank_id) { global_rank_id_ = global_rank_id; }
  // Set global rank size.
  void set_global_rank_size(uint32_t global_rank_size) { global_rank_size_ = global_rank_size; }

  // Set whether need reinitialize collective communication.
  void set_need_reinit_collective(bool need_reinit_collective) { need_reinit_collective_ = need_reinit_collective; }
  // Get whether need reinitialize collective communication.
  bool need_reinit_collective() const { return need_reinit_collective_.load(); }

 private:
  inline static std::shared_ptr<RecoveryContext> instance_{};

  RecoveryContext() = default;
  DISABLE_COPY_AND_ASSIGN(RecoveryContext);

  // Initialize recovery context.
  void Initialize();

  // Create persitent json file, used to persist recovery config.
  void CreatePersistentFile();

  // Obtain the step corresponding to the local latest checkpoint in each training process.
  void ObtainLocalLatestCkptInfo();

  // Parse latest epoch and step info from all latest checkpoints info allgather from other workers.
  void ParseLatestCkptInfo(const int *recv_buffer, const uint32_t buffer_len);

  // Whether enable recovery or not, set by environment variable 'MS_ENABLE_RECOVERY'.
  bool enable_recovery_{false};

  // The persistent directory, set by environment variable 'MS_RECOVERY_PATH'.
  std::string recovery_path_;

  // The interval to persist model, default value: 30 second. set by environment variable 'MS_RECOVERY_INTERVAL'.
  int recovery_interval_{30};

  // Local checkpoint file list.
  std::vector<std::string> ckpt_files_;
  // The file name of latest checkpoint.
  std::string latest_ckpt_file_;
  // The epoch of latest checkpoint.
  int latest_ckpt_epoch_{-1};
  // The step of latest checkpoint.
  int latest_ckpt_step_{-1};

  // Node role in cluster, could be 'MS_WORKER', 'MS_SERVER' or 'MS_SCHED'.
  std::string node_role_;

  // The global rank id of this process. Normally this range is 0 to `global_rank_size_ - 1`.
  uint32_t global_rank_id_{0};
  // The global rank size.
  uint32_t global_rank_size_{0};

  // Whether need to reset training process or not.
  bool need_reset_{false};

  // Whether need to sync the weight of model to device, this value needs to be set to true when python layer
  // performs load checkpoint.
  bool need_sync_weight_to_device_{false};

  // Whether need reinitialize collective communication, this value should be set to true once a training process
  // exits unexpectedly is detected.
  std::atomic_bool need_reinit_collective_{false};

  // Whether the recovery context is already initialized.
  bool initialized_{false};

  // The error status of recovery.
  RecoveryErrCode recovery_status_{RecoveryErrCode::kUnKnownError};

  // The persitent json file util, used to persist recovery config.
  std::unique_ptr<JsonUtils> persistent_json_;
};
}  // namespace recovery
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_RECOVERY_RECOVERY_H_
