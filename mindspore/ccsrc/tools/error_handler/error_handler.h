/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
#define MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "tools/visible.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"

using FuncGetRecentErrMsg = std::function<const char *()>;

namespace mindspore {
namespace tools {
struct FuncInfo {
  const char *caller_file;
  int caller_line;
  const char *caller_func;
  std::string api_msg;
};

enum class ErrorType : int {
  kNoneError = 0,
  kDeviceMemError,
  kHbmMultBitEccError,
  kCommOpRetryFailError,
  kForceStopError,
  kSuspectRemoteError,
  kUnknownError
};

class TOOLS_EXPORT ErrorHandler {
 public:
  static ErrorHandler &GetInstance();

  virtual ~ErrorHandler() = default;
  DISABLE_COPY_AND_ASSIGN(ErrorHandler)

  // Send and receive parameters, return 0 when success, otherwise return 1
  int SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank);

  std::vector<uint64_t> GetOptimizerTimestamps();

  void TftCheckBeforeGraphRun();

  void TftProcessGraphRunError(const std::function<void()> &fn_reset_actor_state,
                               const std::function<void()> &fn_reset_actor_set_state);

  void ProcessError(const FuncInfo &fn_info, int error_code, const FuncGetRecentErrMsg &fn_get_recent_err_msg,
                    ErrorType error_type, bool throw_exception = false);

  const char *GetErrorMsg() const {
    if (error_type_ == ErrorType::kDeviceMemError) {
      return "UCEError error occurs when execute, error_code=507053";
    } else if (error_type_ == ErrorType::kHbmMultBitEccError) {
      return "UCEError error occurs when execute, error_code=507054";
    } else if (error_type_ == ErrorType::kCommOpRetryFailError) {
      return "HCCEError error occurs when execute";
    } else if (error_type_ == ErrorType::kSuspectRemoteError) {
      return "SuspectRemoteError error occurs when execute";
    } else if (error_type_ == ErrorType::kNoneError) {
      return "No uce error occurs.";
    } else {
      return "Unknown error occurs.";
    }
  }

  const char *GetForceStopErrorMsg() const { return "ForceStopError error occurs when execute"; }

  void SaveConstants(const std::vector<KernelGraphPtr> &graphs);
  const ValuePtr &GetConstant(const AnfNodePtr &node);
  void Clear();

  bool GetForceStopFlag() const { return force_stop_flag_; }
  void SetForceStopFlag(bool val) { force_stop_flag_ = val; }

  static uint64_t ExtractUceTime(const char *error_msg);
  bool HasThrownError() const { return force_stop_flag_ || GetUceFlag() || is_reboot_node_ || GetSuspectRemoteFlag(); }

  bool GetUceFlag() const {
    return error_type_ == ErrorType::kDeviceMemError || error_type_ == ErrorType::kHbmMultBitEccError;
  }
  bool GetHcceFlag() const { return error_type_ == ErrorType::kCommOpRetryFailError; }
  bool GetSuspectRemoteFlag() const { return error_type_ == ErrorType::kSuspectRemoteError; }
  void ClearErrorType() { error_type_ = ErrorType::kNoneError; }

  void SetRebootNode(bool flag) { is_reboot_node_ = flag; }
  bool IsRebootNode() const { return is_reboot_node_; }
  void SetIsArf(bool flag) { is_arf_ = flag; }
  bool IsArf() const { return is_arf_; }
  void SetRebuildGroupFlag(bool flag) { rebuild_group_ = flag; }
  bool GetRebuildGroupFlag() const { return rebuild_group_; }
  void SetUceOccurTime(uint64_t time) { uce_occur_time_ = time; }
  void SetRebootType(const std::string &type) { reboot_type_ = type; }
  const std::string &GetRebootType() const { return reboot_type_; }
  uint64_t GetUceOccurTime() { return uce_occur_time_; }

 private:
  // singleton, make constructor private
  ErrorHandler() = default;

  // save constant values for uce scenario, for constant tensor device memory may be corrupted
  std::map<AnfNodePtr, ValuePtr> const_values_{};

  bool force_stop_flag_{false};
  bool is_reboot_node_{false};
  bool is_arf_{false};
  bool rebuild_group_{false};
  std::string reboot_type_{""};
  uint64_t uce_occur_time_{0};
  ErrorType error_type_{ErrorType::kNoneError};
  bool is_graph_pipeline_compiled_{false};
};

using ErrorHandlerPtr = std::shared_ptr<ErrorHandler>;

// Parameter snapshot manager
class TOOLS_EXPORT SnapshotMgr {
 public:
  static SnapshotMgr &GetInstance();

  ~SnapshotMgr() = default;
  DISABLE_COPY_AND_ASSIGN(SnapshotMgr)

  void SaveParameters(const std::vector<AnfNodePtr> &weights, void *stream, device::DeviceResManager *res_manager);

  bool IsSavingSnapshot() const { return is_saving_snapshot_; }
  void SetSavingSnapshot(bool val) { is_saving_snapshot_ = val; }

  std::map<std::string, tensor::TensorPtr> &GetSavedParams() { return saved_params_; }

  int LastSaveStep() const { return last_save_step_; }
  void SaveLastSaveStep(int val) { last_save_step_ = val; }

  bool IsSnapshotValid() { return last_save_step_ > 0; }

  void Reset() {
    last_save_step_ = 0;
    saved_params_.clear();
  }

 private:
  // singleton, make constructor private
  SnapshotMgr() = default;

  // whether is in the progress of copying parameters from device to host
  bool is_saving_snapshot_ = false;

  std::map<std::string, tensor::TensorPtr> saved_params_;
  int last_save_step_ = 0;
};

using SnapshotMgrPtr = std::shared_ptr<SnapshotMgr>;

TOOLS_EXPORT void TftResetOptimizerEventInfo();
}  // namespace tools
}  // namespace mindspore

#endif  // MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_HANDLER_H_
