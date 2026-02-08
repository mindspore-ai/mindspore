/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>

#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "tools/error_handler/exit_handler.h"
#include "include/utils/tensor_py.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "backend/ms_backend/runtime/graph_executor/pre_launch/pre_launch_comm.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "ir/tensor_new.h"
#include "tools/error_handler/error_handler.h"
#include "tools/error_handler/error_config.h"
#include "include/utils/callback.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace mindspore {
using DeviceContext = mindspore::device::DeviceContext;
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
using ParameterStore = mindspore::runtime::ParameterStore;
using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
using mindspore::tools::ErrorHandler;
using mindspore::tools::SnapshotMgr;
namespace {
DeviceContextPtr GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_name << " is not created yet.";
  }
  return device_ctx;
}

constexpr auto RS_NORMAL = "RS_NORMAL";
constexpr auto RS_UCE_HIGHLEVEL = "RS_UCE_HIGHLEVEL";
constexpr auto RS_UCE_LOWLEVEL = "RS_UCE_LOWLEVEL";
constexpr auto RS_UNKNOWN = "RS_UNKNOWN";
}  // namespace

bool GetMemUceInfo(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  return device_ctx->device_res_manager_->GetMemUceInfo(device_id);
}

std::vector<uint64_t> GetOptimizerTimestamps() { return ErrorHandler::GetInstance().GetOptimizerTimestamps(); }

bool GetUceLevelWithMemPoolForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                  const DeviceMemInfo &common_mem_blocks_info,
                                  const std::vector<std::pair<device::DeviceMemPtr, size_t>> &mem_uce_addr) {
  for (auto iter = persistent_mem_blocks_info.begin(); iter != persistent_mem_blocks_info.end(); ++iter) {
    void *persistent_block_start_addr = iter->first;
    auto block_info = iter->second.begin();
    void *persistent_block_end_addr = reinterpret_cast<char *>(persistent_block_start_addr) + block_info->second - 1;
    MS_EXCEPTION_IF_NULL(persistent_block_end_addr);
    for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
      void *mem_uce_start_addr = mem_uce_addr[i].first;
      void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
      MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
      if ((persistent_block_end_addr >= mem_uce_start_addr && persistent_block_start_addr <= mem_uce_start_addr) ||
          (mem_uce_end_addr >= persistent_block_start_addr && mem_uce_start_addr <= persistent_block_start_addr)) {
        MS_LOG(INFO) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }

  for (auto iter = common_mem_blocks_info.begin(); iter != common_mem_blocks_info.end(); ++iter) {
    void *common_block_start_addr = iter->first;
    auto block_info = iter->second.begin();
    void *common_block_end_addr = reinterpret_cast<char *>(common_block_start_addr) + block_info->second - 1;
    MS_EXCEPTION_IF_NULL(common_block_end_addr);
    for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
      void *mem_uce_start_addr = mem_uce_addr[i].first;
      void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
      MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
      if ((common_block_end_addr >= mem_uce_start_addr && common_block_start_addr <= mem_uce_start_addr) ||
          (mem_uce_end_addr >= common_block_start_addr && mem_uce_start_addr <= common_block_start_addr)) {
        MS_LOG(INFO) << "UCE process strategy is RS_UCE_LOWLEVEL.";
        return true;
      }
    }
  }
  return false;
}

std::string GetUceProcessStrategyForKbk(const DeviceMemInfo &persistent_mem_blocks_info,
                                        const DeviceMemInfo &common_mem_blocks_info,
                                        const std::vector<std::pair<device::DeviceMemPtr, size_t>> &mem_uce_addr) {
  // Judge whether weights got uce error.
  MS_LOG(INFO) << "Start to get UCE process strategy for kbk.";
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  const auto &kernel_tensors_with_info = graph_parameter_store->GetAll();
  try {
    for (size_t outer_idx = 0; outer_idx < kernel_tensors_with_info.size(); ++outer_idx) {
      if (!graph_parameter_store->GetPositionWeight(outer_idx)) {
        continue;
      }
      auto kernel_tensor_with_info = kernel_tensors_with_info[outer_idx];
      for (size_t inner_idx = 0; inner_idx < kernel_tensor_with_info.size(); ++inner_idx) {
        auto kernel_tensor = kernel_tensor_with_info[inner_idx].first;
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        const auto &device_tensor = kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(device_tensor);
        void *device_tensor_start_addr = const_cast<void *>(device_tensor->GetPtr());
        void *device_tensor_end_addr =
          reinterpret_cast<char *>(device_tensor_start_addr) + device_tensor->GetSize() - 1;
        MS_EXCEPTION_IF_NULL(device_tensor_end_addr);
        for (size_t i = 0; i < mem_uce_addr.size(); ++i) {
          void *mem_uce_start_addr = mem_uce_addr[i].first;
          void *mem_uce_end_addr = reinterpret_cast<char *>(mem_uce_start_addr) + mem_uce_addr[i].second - 1;
          MS_EXCEPTION_IF_NULL(mem_uce_end_addr);
          // Return RS_UCE_HIGHLEVEL if overlap of device tensor addr and mem uce addr.
          if ((device_tensor_end_addr >= mem_uce_start_addr && device_tensor_start_addr <= mem_uce_start_addr) ||
              (mem_uce_end_addr >= device_tensor_start_addr && mem_uce_start_addr <= device_tensor_start_addr)) {
            MS_LOG(INFO) << "UCE process strategy is RS_UCE_HIGHLEVEL.";
            return RS_UCE_HIGHLEVEL;
          }
        }
      }
    }

    // Return RS_UCE_LOWLEVEL if overlap of memory pool addr and mem uce addr.
    if (GetUceLevelWithMemPoolForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_addr)) {
      return RS_UCE_LOWLEVEL;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "There is an error: " << e.what();
  }

  MS_LOG(INFO) << "UCE process strategy is RS_NORMAL.";

  return RS_NORMAL;
}

std::string GetUceProcessStrategy() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto persistent_mem_blocks_info = device_ctx->device_res_manager_->GetPersistentMemBlocksInfoStatistics();
  auto common_mem_blocks_info = device_ctx->device_res_manager_->GetCommonMemBlocksInfoStatistics();
  auto mem_uce_addr = device_ctx->device_res_manager_->GetMemUceAddr();
  return GetUceProcessStrategyForKbk(persistent_mem_blocks_info, common_mem_blocks_info, mem_uce_addr);
}

void UceMemRepair(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  device_ctx->device_res_manager_->UceMemRepair(device_id);
}

void StopDevice(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  MS_LOG(WARNING) << "Try to stop device: " << device_id;
  ErrorHandler::GetInstance().SetForceStopFlag(true);
  device_ctx->device_res_manager_->StopDevice(device_id);
  MS_LOG(WARNING) << "stop device: " << device_id << " end;";
}

int CleanTdtChannel() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  device_ctx->Initialize();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->GetKernelExecutor());
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  return device_ctx->GetKernelExecutor()->CleanTdtChannel();
}

void FinalizeCommunication() {
  MS_LOG(WARNING) << "Try to finalize communication";
  auto group_info = distributed::collective::CollectiveManager::instance()->get_group_info();
  for (const auto &item : group_info) {
    MS_LOG(WARNING) << "Destroy group, group name: " << item.first << ", ranks: " << item.second;
    if (!distributed::collective::CollectiveManager::instance()->DestroyDeviceSideCommunicationGroup(item.first)) {
      MS_LOG(EXCEPTION) << "Destroy group:" << item.first << " failed, ranks: " << item.second;
    }
    MS_LOG(WARNING) << "Destroy group, group name: " << item.first << " ok";
  }
  MS_LOG(WARNING) << "Finalize communication end";
}

void RebuildGroup() {
  auto &error_handler = ErrorHandler::GetInstance();
  // rebuild comm
  MS_LOG(WARNING) << "Try to rebuild group communication";
  error_handler.SetRebuildGroupFlag(true);
  auto group_info = distributed::collective::CollectiveManager::instance()->get_group_info();
  distributed::collective::CollectiveManager::instance()->ClearInitResult();
  device::GroupOptions config = {};
  config.async = true;
  for (const auto &item : group_info) {
    MS_LOG(WARNING) << "Rebuild group, group name: " << item.first << ", ranks: " << item.second;
    if (!distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(item.first, item.second,
                                                                                          config)) {
      MS_LOG(EXCEPTION) << "Rebuild group:" << item.first << " failed, ranks: " << item.second;
    }
    MS_LOG(WARNING) << "Rebuild group, group name: " << item.first << " ok";
  }
  (void)distributed::collective::CollectiveManager::instance()->WaitAllCommInitDone();
  MS_LOG(WARNING) << "All group init done";
  error_handler.ClearErrorType();
  MS_LOG(WARNING) << "Rebuild communication end";
}
bool IsRebootNode() { return ErrorHandler::GetInstance().IsRebootNode(); }

void SetIsRebootNode(bool is_reboot) {
  MS_LOG(WARNING) << "Set is reboot node flag: " << is_reboot;
  ErrorHandler::GetInstance().SetRebootNode(is_reboot);
}

void SetRebootNodeType(const std::string &type) {
  MS_LOG(WARNING) << "Set is reboot node reboot type: " << type;
  ErrorHandler::GetInstance().SetRebootType(type);
}

string GetRebootNodeType() { return ErrorHandler::GetInstance().GetRebootType(); }

void SetIsArf(bool is_arf) {
  auto &error_handler = ErrorHandler::GetInstance();
  MS_LOG(WARNING) << "Set is arf flag: " << is_arf;
  error_handler.SetIsArf(is_arf);
  if (!is_arf) {
    // reset reboot node flag when reset arf flag at train step end
    error_handler.SetRebootNode(false);
    error_handler.SetRebootType("");
    error_handler.SetRebuildGroupFlag(false);
  }
}

bool GetIsArf() { return ErrorHandler::GetInstance().IsArf(); }

void ResetErrorState() { ErrorHandler::GetInstance().SetForceStopFlag(false); }

void RePreLaunchSendRecv(int32_t device_id) {
  MS_LOG(WARNING) << "Try to pre-launch send recv. device id: " << device_id;
  static auto disable_pre_build_comm = runtime::IsDisableRuntimeConfig(runtime::kRuntimePreBuildCommKernel);
  if (disable_pre_build_comm) {
    return;
  }
  const auto &launch_orders = runtime::PreLaunchComm::GetInstance().GetPreLaunchOrder(true);
  for (auto graph_id : launch_orders) {
    const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_id);
    MS_EXCEPTION_IF_NULL(actor_set);
    PROF_START(PreLaunchCommKernel);
    runtime::PreLaunchComm::GetInstance().PreLaunchCommKernel(actor_set);
    PROF_END(PreLaunchCommKernel);
  }
  MS_LOG(WARNING) << "Pre-launch send recv success";
}

int RegSnapshotParams(const std::map<std::string, py::object> &param_dict) {
  auto &mem_ckpt_params = SnapshotMgr::GetInstance().GetSavedParams();
  if (!mem_ckpt_params.empty()) {
    // parameters has already been registered
    MS_LOG(INFO) << "Parameters has already been registered.";
    return 1;
  }
  for (auto &[name, param] : param_dict) {
    mem_ckpt_params[name] = nullptr;
    auto tensor = tensor::ConvertToTensor(param);
    MS_ERROR_IF_NULL(tensor);
    MS_LOG(DEBUG) << name << " shape: " << tensor->shape_c() << " size: " << tensor->Size();
  }

  MS_LOG(INFO) << "Parameters has been registered successfully.";
  return 0;
}

void ResetSnapshotState() { SnapshotMgr::GetInstance().Reset(); }

void ClearSnapshotSavingFlag() { SnapshotMgr::GetInstance().SetSavingSnapshot(false); }

bool IsSnapshotValid() { return SnapshotMgr::GetInstance().IsSnapshotValid(); }

std::map<std::string, py::object> GetSnapshotParams() {
  auto &snapshot_mgr = SnapshotMgr::GetInstance();
  // if parameter snapshot has not been generated, return an empty map
  if (!snapshot_mgr.IsSnapshotValid()) {
    return std::map<std::string, py::object>();
  }

  std::map<std::string, py::object> param_dict;
  for (auto &[name, tensor] : snapshot_mgr.GetSavedParams()) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Value of parameter " << name << " is null, skip it.";
      continue;
    }
    param_dict[name] = tensor::PackTensorToPyObject(tensor);
  }

  // append step_num to param_dict for resuming training
  constexpr char kStepNum[] = "step_num";
  // since snapshot was saved before optimizer, so here minus 1
  int step_num = snapshot_mgr.LastSaveStep() - 1;
  auto tensor = tensor::from_buffer(kNumberTypeInt32, ShapeVector{}, &step_num, sizeof(step_num));
  param_dict[kStepNum] = tensor::PackTensorToPyObject(tensor);

  return param_dict;
}

void RegisterConfig(const py::object &configs) {
  MS_EXCEPTION_IF_NULL(tools::TftConfig::GetInstance());
  tools::TftConfig::GetInstance()->RegisterConfig(configs);
}

void RegTFT(py::module *m) {
  (void)m->def("_stop_device", &mindspore::StopDevice, "Stop the device.");
  (void)m->def("clean_tdt_channel", &mindspore::CleanTdtChannel, "Clean tdt channel");
  (void)m->def("_repair_device", &mindspore::UceMemRepair, "Repair the device.");
  (void)m->def("_get_uce_process_strategy", &mindspore::GetUceProcessStrategy, "Get UCE process strategy.");
  (void)m->def("_get_uce_mem_info", &mindspore::GetMemUceInfo, "Get UCE mem info.");
  (void)m->def("_get_optimzer_timestamps", &mindspore::GetOptimizerTimestamps,
               "Get optimizer start and finish timestamps.");
  (void)m->def("_tft_sem_post", []() { mindspore::tools::TFTWaitSem::GetInstance().Post(); }, "TFT sem start post");
  (void)m->def("_tft_sem_enable", []() { mindspore::tools::TFTWaitSem::Enable(); }, "TFT enable sem feature");
  (void)m->def(
    "_tft_start_record_threads", []() { mindspore::tools::TFTWaitSem::GetInstance().StartRecordThreads(); },
    "TFT start recording newly created threads");
  (void)m->def(
    "_tft_finish_record_threads", []() { mindspore::tools::TFTWaitSem::GetInstance().FinishRecordThreads(); },
    "TFT finish recording newly created threads");
  (void)m->def("_finalize_comm", &FinalizeCommunication, "Finalize comm.");
  (void)m->def("_rebuild_group", &RebuildGroup, "Rebuild group");
  (void)m->def("is_reboot_node", &IsRebootNode, "Get reboot node flag.");
  (void)m->def("_pre_launch_send_recv", &RePreLaunchSendRecv, "PreLaunch Send Recv before launch graph.");
  (void)m->def("set_is_reboot_node", &SetIsRebootNode, "Set reboot node flag for arf.");
  (void)m->def("set_reboot_type", &SetRebootNodeType, "Set reboot node type, arf or hot_switch.");
  (void)m->def("get_reboot_type", &GetRebootNodeType, "Get reboot node type, arf or hot_switch.");
  (void)m->def("check_is_arf", &GetIsArf, "Get arf flag.");
  (void)m->def("set_is_arf", &SetIsArf, "Set arf flag.");
  (void)m->def("_reg_snapshot_params", &mindspore::RegSnapshotParams, "Register parameters for snapshot",
               py::arg("param_dict"));
  (void)m->def("_reset_snapshot_state", &mindspore::ResetSnapshotState, "Reset snapshot state");
  (void)m->def("_is_snapshot_valid", &mindspore::IsSnapshotValid,
               "Return true when snapshot is valid, otherwise false.");
  (void)m->def("_clear_snapshot_saving_flag", &mindspore::ClearSnapshotSavingFlag, "Clear snapshot saving flag.");
  (void)m->def("_get_snapshot_params", &mindspore::GetSnapshotParams, "Get parameters from snapshot");
  (void)m->def("tft_register_config", &RegisterConfig, "Register all configs.");
  (void)m->def("_reset_error_state", &ResetErrorState, "Reset error state of ErrorHandler.");
  (void)m->def("_reset_opt_event_info", &mindspore::tools::TftResetOptimizerEventInfo, "Reset optimizer event info.");
}
}  // namespace mindspore
