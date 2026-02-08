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

#include "tools/error_handler/error_handler.h"
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "error_handler/error_config.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/utils/callback.h"
#include "device_address/device_type.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace tools {
using mindspore::device::DeviceContext;
using mindspore::device::DeviceResManager;
using mindspore::device::DeviceType;
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;

namespace {
constexpr char kStrUceTimeBegin[] = "time us=";
constexpr size_t kStrUceTimeBeginLen = sizeof(kStrUceTimeBegin) - 1;
constexpr int64_t kInterval = 30;

bool NeedSaveAsyncCkpt() {
  static bool disable_ckpt_d2h_async = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC") != "1";
  if (MS_LIKELY(disable_ckpt_d2h_async)) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_NEED_CKPT)) {
    return false;
  }

  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
  auto checkpoint_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", checkpoint_steps: " << checkpoint_steps
                << ", last_triggered_step:" << last_triggered_step;
  return cur_step >= (last_triggered_step + checkpoint_steps);
}

bool NeedSaveSnapshot() {
  if (!TftConfig::IsEnableStepTRE()) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto last_save_step = SnapshotMgr::GetInstance().LastSaveStep();
  auto snapshot_steps = TftConfig::GetSnapShotSteps();
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", snapshot_steps: " << snapshot_steps
                << ", last_save_step:" << last_save_step;
  return last_save_step > 0 ? cur_step >= (last_save_step + snapshot_steps) : cur_step > snapshot_steps;
}

ErrorType GetErrorType(int error_code) {
  static std::vector<std::pair<ErrorType, std::function<bool(int)>>> error_type_callbacks = {
    {ErrorType::kDeviceMemError, GET_COMMON_CALLBACK(IsDeviceMemError, bool, int)},
    {ErrorType::kHbmMultBitEccError, GET_COMMON_CALLBACK(IsHbmMultBitEccError, bool, int)},
    {ErrorType::kCommOpRetryFailError, GET_COMMON_CALLBACK(IsCommOpRetryFailError, bool, int)},
    {ErrorType::kForceStopError, GET_COMMON_CALLBACK(IsForceStopError, bool, int)},
    {ErrorType::kSuspectRemoteError, GET_COMMON_CALLBACK(IsSuspectRemoteError, bool, int)},
  };

  for (auto &[error_type, func] : error_type_callbacks) {
    if (func != nullptr && func(error_code)) {
      return error_type;
    }
  }
  return ErrorType::kUnknownError;
}

}  // namespace

ErrorHandler &ErrorHandler::GetInstance() {
  static ErrorHandler instance;
  return instance;
}

int ErrorHandler::SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank) {
  static auto send_recv_cb =
    GET_COMMON_CALLBACK(TftSendRecvParams, int, const std::vector<tensor::TensorPtr> &, int, int, bool);
  if (send_recv_cb == nullptr) {
    return 1;
  }
  return send_recv_cb(params, src_rank, dst_rank, !is_arf_);
}

std::vector<uint64_t> ErrorHandler::GetOptimizerTimestamps() {
  static auto get_opt_times_cb = GET_COMMON_CALLBACK(TftGetOptimizerTimestamps, std::pair<uint64_t, uint64_t>);
  MS_EXCEPTION_IF_NULL(get_opt_times_cb);
  auto [opt_start_timestamp, opt_end_timestamp] = get_opt_times_cb();
  auto hbm_error_time = GetUceOccurTime();
  return std::vector<uint64_t>{hbm_error_time, opt_start_timestamp, opt_end_timestamp};
}

void ErrorHandler::TftCheckBeforeGraphRun() {
  if (tools::TftConfig::GetInstance()->IsEnableUCE() || tools::TftConfig::GetInstance()->IsEnableHCCE() ||
      tools::TftConfig::GetInstance()->IsEnableARF()) {
    if (GetHcceFlag()) {
      MS_LOG(INFO) << "Restart from step after a hcce error occurs.";
    } else if (GetSuspectRemoteFlag()) {
      MS_LOG(INFO) << "Restart from step after a SuspectRemote error occurs.";
    } else if (GetUceFlag()) {
      MS_LOG(INFO) << "Restart from step after a uce error occurs.";
    } else if (GetForceStopFlag()) {
      MS_LOG(EXCEPTION) << "ForceStopError occurs when execute.";
    }
  }
}

void ErrorHandler::TftProcessGraphRunError(const std::function<void()> &fn_reset_actor_state,
                                           const std::function<void()> &fn_reset_actor_set_state) {
  if (!(tools::TftConfig::GetInstance()->IsEnableUCE() || tools::TftConfig::GetInstance()->IsEnableHCCE() ||
        tools::TftConfig::GetInstance()->IsEnableARF())) {
    return;
  }

  if (HasThrownError()) {
    if (GetForceStopFlag()) {
      MS_LOG(WARNING) << "There is a ForceStop error, reset the actor state.";
    }
    if (GetHcceFlag()) {
      MS_LOG(WARNING) << "There is a HCCE error, reset the actor state.";
    } else if (GetSuspectRemoteFlag()) {
      MS_LOG(WARNING) << "There is a SuspectRemote error, reset the actor state.";
    } else if (GetUceFlag()) {
      MS_LOG(WARNING) << "There is a UCE error, reset the actor state.";
    }
    // reset actor state
    fn_reset_actor_state();
    MsException::Instance().ResetException();
    MS_LOG(WARNING) << "Clear state end.";
  }

  if (GetUceFlag()) {
    MS_LOG(EXCEPTION) << GetErrorMsg();
  } else if (GetForceStopFlag()) {
    fn_reset_actor_set_state();
    MS_LOG(EXCEPTION) << GetForceStopErrorMsg();
  }
}

void ErrorHandler::ProcessError(const FuncInfo &fn_info, int error_code,
                                const FuncGetRecentErrMsg &fn_get_recent_err_msg, ErrorType error_type,
                                bool throw_exception) {
  const std::string &api_func = fn_info.api_msg;
  if (api_func == "aclrtProcessReport" || api_func == "acltdtReceiveTensor" || api_func == "aclDestroyDataBuffer") {
    MS_LOG(DEBUG) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  } else {
    MS_LOG(ERROR) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  }

  switch (error_type) {
    case ErrorType::kHbmMultBitEccError: {
      if (fn_get_recent_err_msg != nullptr && !HasThrownError()) {
        SetUceOccurTime(ExtractUceTime(fn_get_recent_err_msg()));
      }
    }
    case ErrorType::kDeviceMemError:
    case ErrorType::kCommOpRetryFailError:
    case ErrorType::kSuspectRemoteError: {
      if (!HasThrownError()) {
        error_type_ = error_type;
        if (throw_exception) {
          MS_LOG(EXCEPTION) << GetErrorMsg() << ". Error code is " << error_code;
        } else {
          MS_LOG(ERROR) << GetErrorMsg() << ". Error code is " << error_code;
        }
      }
      break;
    }
    case ErrorType::kForceStopError: {
      SetForceStopFlag(true);
      MS_LOG(ERROR) << "ForceStopError error occurs when execute";
      break;
    }
    default: {
      break;
    }
  }
}

// extract UCE occurs time from string "HBM MULTI BIT ECC, Uncorrectable ECC, device_id=3,
// event_id=0x80e01801, time us=67672363666.[FUNC:ProcHBMRas][FILE:stars_engine.cc]"
uint64_t ErrorHandler::ExtractUceTime(const char *error_msg) {
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Error message is " << error_msg;
  if (error_msg == nullptr) {
    return 0;
  }
  std::string message = error_msg;
  auto idx_begin = message.find(kStrUceTimeBegin);
  if (idx_begin == std::string::npos) {
    return 0;
  }
  size_t num_digits = 0;
  for (auto idx = idx_begin + kStrUceTimeBeginLen; idx < message.size(); ++idx) {
    if (!isdigit(message[idx])) {
      break;
    }
    num_digits += 1;
  }
  if (num_digits == 0) {
    return 0;
  }
  auto decimal_str = message.substr(idx_begin + kStrUceTimeBeginLen, num_digits);
  try {
    auto time_us = std::stoull(decimal_str);
    MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Extracted time is " << time_us << " us.";
    return time_us;
  } catch (std::logic_error const &ex) {
    MS_LOG(ERROR) << "Convert decimal string " << decimal_str << " to uint64_t value failed.";
    return 0;
  }
}

void ErrorHandler::SaveConstants(const std::vector<KernelGraphPtr> &graphs) {
  if (!TftConfig::GetInstance()->IsEnableUCE()) {
    MS_LOG(INFO) << "Not enable UCE, skip saving constants.";
    return;
  }
  MS_LOG(INFO) << "Save constants of graphs for UCE recovery";
  for (auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    for (const auto &value_node : graph->graph_value_nodes()) {
      auto node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);
      if (node_value->isa<tensor::Tensor>()) {
        auto tensor = node_value->cast<tensor::TensorPtr>();
        const_values_[value_node] = std::make_shared<tensor::Tensor>(*tensor);
      } else {
        const_values_[value_node] = value_node->value();
      }
    }
  }
}

const ValuePtr &ErrorHandler::GetConstant(const AnfNodePtr &node) {
  auto iter = const_values_.find(node);
  if (iter == const_values_.end()) {
    MS_LOG(EXCEPTION) << "Can not find tensor for node " << node->fullname_with_scope();
  }
  return iter->second;
}

void ErrorHandler::Clear() { const_values_.clear(); }

SnapshotMgr &SnapshotMgr::GetInstance() {
  static SnapshotMgr instance;
  return instance;
}

void SnapshotMgr::SaveParameters(const std::vector<AnfNodePtr> &weights, void *stream, DeviceResManager *res_manager) {
  static auto save_params_cb = GET_COMMON_CALLBACK(TftSaveParameters, void, const std::vector<AnfNodePtr> &, void *,
                                                   std::map<std::string, tensor::TensorPtr> *);
  if (save_params_cb != nullptr) {
    save_params_cb(weights, res_manager->GetCopyDataStream(), &saved_params_);
  }
}

void DestroySnapshotMgr() { SnapshotMgr::GetInstance().Reset(); }

REGISTER_COMMON_CALLBACK(DestroySnapshotMgr);

void TftCheckBeforeGraphRun() { ErrorHandler::GetInstance().TftCheckBeforeGraphRun(); }

void TftProcessGraphRunError(const std::function<void()> &fn_reset_actor_state,
                             const std::function<void()> &fn_reset_actor_set_state) {
  ErrorHandler::GetInstance().TftProcessGraphRunError(fn_reset_actor_state, fn_reset_actor_set_state);
}

void TftSaveConstants(const std::vector<KernelGraphPtr> &graphs) { ErrorHandler::GetInstance().SaveConstants(graphs); }

bool SkipHcomInitWait() {
  auto reboot_type = ErrorHandler::GetInstance().GetRebootType();
  auto rebuild_flag = ErrorHandler::GetInstance().GetRebuildGroupFlag();
  return reboot_type == "hot_switch" && !rebuild_flag;
}

bool SkipSubmitTask() {
  auto reboot_type = ErrorHandler::GetInstance().GetRebootType();
  auto rebuild_flag = ErrorHandler::GetInstance().GetRebuildGroupFlag();
  auto flag = reboot_type == "hot_switch" && !rebuild_flag;
  if (flag) {
    MS_LOG(WARNING) << "HOT Switch node no need submit hcom init task before rebuild hcom flag";
  }
  return flag;
}

bool TftGetUceFlag() { return ErrorHandler::GetInstance().GetUceFlag(); }

bool TftIsRebootNode() { return ErrorHandler::GetInstance().IsRebootNode(); }

bool TftIsArf() { return ErrorHandler::GetInstance().IsArf(); }

void TftClearErrorType() { ErrorHandler::GetInstance().ClearErrorType(); }

const ValuePtr &TftGetConstant(const AnfNodePtr &node) { return ErrorHandler::GetInstance().GetConstant(node); }

REGISTER_COMMON_CALLBACK(TftCheckBeforeGraphRun);
REGISTER_COMMON_CALLBACK(TftProcessGraphRunError);
REGISTER_COMMON_CALLBACK(TftSaveConstants);

REGISTER_COMMON_CALLBACK(SkipHcomInitWait);
REGISTER_COMMON_CALLBACK(SkipSubmitTask);

REGISTER_COMMON_CALLBACK(TftGetUceFlag);
REGISTER_COMMON_CALLBACK(TftIsRebootNode);
REGISTER_COMMON_CALLBACK(TftIsArf);
REGISTER_COMMON_CALLBACK(TftClearErrorType);
REGISTER_COMMON_CALLBACK(TftGetConstant);

void RunFailCallback(const char *caller_file, int caller_line, const char *caller_name, const std::string &api_info,
                     bool throw_exception) {
  static auto get_last_error_cb = GET_COMMON_CALLBACK(TftGetErrorCode, int);
  static auto get_recent_err_msg_cb = GET_COMMON_CALLBACK(TftGetRecentErrMsg, const char *);
  if (get_last_error_cb != nullptr &&
      (TftConfig::GetInstance()->IsEnableUCE() || TftConfig::GetInstance()->IsEnableHCCE())) {
    auto error_code = get_last_error_cb();
    auto error_type = GetErrorType(error_code);
    ErrorHandler::GetInstance().ProcessError(FuncInfo{caller_file, caller_line, caller_name, api_info}, error_code,
                                             get_recent_err_msg_cb, error_type, throw_exception);
  }
  if (TftConfig::GetInstance()->IsEnableARF()) {
    if (get_last_error_cb != nullptr) {
      auto error_code = get_last_error_cb();
      auto error_type = GetErrorType(error_code);
      MS_LOG(DEBUG) << "Call ascend api <" << api_info << "> in <" << caller_name << "> at " << caller_file << ":"
                    << caller_line << " failed, error code [" << error_code << "].";
      if (error_type == ErrorType::kForceStopError) {
        ErrorHandler::GetInstance().SetForceStopFlag(true);
      }
    }
  }
}

void TaskFailCallback(std::string *error_info) {
  MS_EXCEPTION_IF_NULL(error_info);
  if (TftConfig::GetInstance()->IsEnableUCE() || TftConfig::GetInstance()->IsEnableARF()) {
    if (ErrorHandler::GetInstance().GetForceStopFlag() && !ErrorHandler::GetInstance().HasThrownError()) {
      if (error_info->empty()) {
        *error_info = std::string(ErrorHandler::GetInstance().GetForceStopErrorMsg());
        MS_LOG(EXCEPTION) << ErrorHandler::GetInstance().GetForceStopErrorMsg();
      }
    }
    if (ErrorHandler::GetInstance().GetUceFlag() && !ErrorHandler::GetInstance().HasThrownError()) {
      if (error_info->empty()) {
        *error_info = std::string(ErrorHandler::GetInstance().GetErrorMsg());
        MS_LOG(EXCEPTION) << ErrorHandler::GetInstance().GetErrorMsg();
      }
    }
  }
}

void SyncCopyFailCallback() {
  auto &error_handler = ErrorHandler::GetInstance();
  MS_LOG(WARNING) << "Uce flag: " << error_handler.GetUceFlag()
                  << ", force stop flag: " << error_handler.GetForceStopFlag();
  if (error_handler.GetUceFlag()) {
    MS_LOG(EXCEPTION) << "UCEError occurs when execute.";
  } else if (error_handler.GetForceStopFlag()) {
    MS_LOG(EXCEPTION) << "ForceStopError occurs when execute.";
  }
}

REGISTER_COMMON_CALLBACK(RunFailCallback);
REGISTER_COMMON_CALLBACK(TaskFailCallback);
REGISTER_COMMON_CALLBACK(SyncCopyFailCallback);

bool NeedRebuildGroup() { return ErrorHandler::GetInstance().GetRebuildGroupFlag(); }

bool IsRebootNode() { return ErrorHandler::GetInstance().IsRebootNode(); }

REGISTER_COMMON_CALLBACK(NeedRebuildGroup);
REGISTER_COMMON_CALLBACK(IsRebootNode);

class TftCallback {
 public:
  static bool OnLaunchBegin(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                            const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                            const DeviceContext *device_context) {
    AsyncSaveSnapshot(kernel, device_context->device_res_manager_.get());
    return ProcessEvent(kernel, kernel_mod, stream);
  }

  static void OnLaunchEnd(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                          const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                          const DeviceContext *device_context, bool launch_success) {
    if (launch_success) {
      return;
    }
    ProcessResumableError();
    if (!ErrorHandler::GetInstance().HasThrownError()) {
      device_context->device_res_manager_->ResetStreamAndCtx();
    }
  }

 private:
  static void AsyncSaveSnapshot(const CNodePtr &kernel, DeviceResManager *res_manager);
  static bool ProcessEvent(const CNodePtr &kernel, KernelMod *kernel_mod, void *stream);
  static void ProcessResumableError();
};

bool IsEnableTftCallback() {
  static bool enable_d2h_async = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC") == "1";
  static bool enabled = TftConfig::GetInstance()->IsEnableUCE() || TftConfig::GetInstance()->IsEnableARF() ||
                        TftConfig::GetInstance()->IsEnableHCCE() || TftConfig::GetInstance()->IsEnableRsc() ||
                        TftConfig::GetInstance()->IsEnableTRE() || TftConfig::GetInstance()->IsEnableStepTRE() ||
                        enable_d2h_async;
  return enabled;
}

REGISTER_PRE_LAUNCH_CALLBACK(TftPreLaunchKernel, &TftCallback::OnLaunchBegin, IsEnableTftCallback, DeviceType::kAscend,
                             0);

REGISTER_POST_LAUNCH_CALLBACK(TftPostLaunchKernel, &TftCallback::OnLaunchEnd, IsEnableTftCallback, DeviceType::kAscend,
                              0);

bool TftCallback::ProcessEvent(const CNodePtr &kernel, KernelMod *kernel_mod, void *stream) {
  static auto process_opt_event =
    GET_COMMON_CALLBACK(TftProcessOptimizerEvent, bool, const CNodePtr &, KernelMod *, void *, bool, bool);
  MS_EXCEPTION_IF_NULL(process_opt_event);
  return process_opt_event(kernel, kernel_mod, stream, SnapshotMgr::GetInstance().IsSavingSnapshot(),
                           TftConfig::GetInstance()->IsEnableUCE());
}

void TftCallback::AsyncSaveSnapshot(const CNodePtr &kernel, DeviceResManager *res_manager) {
  if (!IsGraphPipelineCompiled()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(kernel);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(kernel->func_graph());
  if (kg == nullptr || !kg->is_graph_run_mode()) {
    MS_LOG(INFO) << "Skip copy, since kernel graph is " << (kg == nullptr ? "null" : "not graph mode");
    return;
  }

  if (!NeedSaveAsyncCkpt() && !NeedSaveSnapshot()) {
    return;
  }

  if (!SnapshotMgr::GetInstance().IsSavingSnapshot()) {
    SnapshotMgr::GetInstance().SetSavingSnapshot(true);
    MS_LOG(INFO) << "Enable async d2h copy";
    SnapshotMgr::GetInstance().SaveParameters(kg->GetRootWeights(), res_manager->GetCopyDataStream(), res_manager);
    SnapshotMgr::GetInstance().SaveLastSaveStep(MsContext::GetInstance()->get_param<int>(MS_CTX_CUR_STEP_NUM));
  }
}

void TftCallback::ProcessResumableError() {
  static auto get_last_error_cb = GET_COMMON_CALLBACK(TftGetErrorCode, int);
  static auto get_recent_err_msg_cb = GET_COMMON_CALLBACK(TftGetRecentErrMsg, const char *);
  if (get_last_error_cb != nullptr &&
      (TftConfig::GetInstance()->IsEnableUCE() || TftConfig::GetInstance()->IsEnableHCCE())) {
    auto error_code = get_last_error_cb();
    auto error_type = GetErrorType(error_code);
    ErrorHandler::GetInstance().ProcessError(FuncInfo{FILE_NAME, __LINE__, __FUNCTION__, "Launch kernel failed"},
                                             error_code, get_recent_err_msg_cb, error_type, false);
  }
  if (TftConfig::GetInstance()->IsEnableARF()) {
    if (get_last_error_cb != nullptr) {
      auto error_code = get_last_error_cb();
      MS_LOG(DEBUG) << "Launch kernel failed in <" << __FUNCTION__ << "> at " << FILE_NAME << ":" << __LINE__
                    << " failed, error code [" << error_code << "].";
      auto error_type = GetErrorType(error_code);
      if (error_type == ErrorType::kForceStopError) {
        ErrorHandler::GetInstance().SetForceStopFlag(true);
      }
    }
  }
}

void InitWatchDogCallback() {
  static auto init_cb =
    GET_COMMON_CALLBACK(TftInitWatchDogCallback, void, bool, bool, bool, int64_t, const std::string &);
  if (init_cb != nullptr) {
    init_cb(TftConfig::GetInstance()->IsEnableWatchdog(), TftConfig::GetInstance()->IsEnableSaveHcclOpStatus(),
            TftConfig::GetInstance()->CheckSupport(kStatusRecord, false),
            TftConfig::GetInstance()->GetConfigValue<int64_t>(kStatusSaveInterval, kInterval),
            TftConfig::GetInstance()->GetConfigValue<std::string>(kStatusSavePath, "/tmp"));
  }
}

bool WatchDogPreLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                             const DeviceContext *device_context) {
  static auto func_cb =
    GET_COMMON_CALLBACK(TftWatchDogPreLaunchKernel, bool, const CNodePtr &, const std::vector<KernelTensor *> &,
                        const std::vector<KernelTensor *> &, KernelMod *, void *, const DeviceContext *);
  static std::once_flag flag;
  std::call_once(flag, [&]() { tools::InitWatchDogCallback(); });
  return func_cb == nullptr ? false : func_cb(kernel, inputs, outputs, kernel_mod, stream, device_context);
}

void WatchDogPostLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                              const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                              const DeviceContext *device_context, bool launch_success) {
  static auto func_cb =
    GET_COMMON_CALLBACK(TftWatchDogPostLaunchKernel, void, const CNodePtr &, const std::vector<KernelTensor *> &,
                        const std::vector<KernelTensor *> &, KernelMod *, void *, const DeviceContext *, bool);
  static std::once_flag flag;
  std::call_once(flag, [&]() { tools::InitWatchDogCallback(); });
  if (func_cb != nullptr) {
    func_cb(kernel, inputs, outputs, kernel_mod, stream, device_context, launch_success);
  }
}

REGISTER_PRE_LAUNCH_CALLBACK(
  WatchDogPreLaunchKernel, WatchDogPreLaunchKernel, []() { return TftConfig::GetInstance()->IsEnableWatchdog(); },
  DeviceType::kAscend, 0);

REGISTER_POST_LAUNCH_CALLBACK(
  WatchDogPostLaunchKernel, WatchDogPostLaunchKernel, []() { return TftConfig::GetInstance()->IsEnableWatchdog(); },
  DeviceType::kAscend, 0);

void TftResetOptimizerEventInfo() {
  static auto reset_cb = GET_COMMON_CALLBACK(ResetOptimizerEventInfo, void);
  if (reset_cb != nullptr) {
    reset_cb();
  }
}
}  // namespace tools
}  // namespace mindspore
