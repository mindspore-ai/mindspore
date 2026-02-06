/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include <dlfcn.h>
#include <map>
#include <algorithm>
#include <sstream>
#include "include/common/utils/utils.h"
#include "symbolic_shape/symbol.h"
#define google ascend_private
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "ge/ge_api_types.h"
#undef google
#include "hccl/hccl.h"
#include "hccl/hcom.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/backend/distributed/constants.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/ascend_op_name.h"
#include "mindspore/ops/op_def/framework_op_name.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

static constexpr const auto kHcclPluginFileName = "libhccl_plugin.so";

#define CHECK_SYMBOL_NULL(symbol)                                                    \
  if ((symbol) == nullptr) {                                                         \
    MS_LOG(WARNING) << #symbol << " is null, hccl has not been inited, do nothing."; \
    return HcclResult::HCCL_E_RESERVED;                                              \
  }

static std::string GenerateCMChiefWorkDevice() {
  if (mindspore::common::GetEnv(mindspore::distributed::kEnvWorkerNum) == "1") {
    // If only one device is used, use 'DEVICE_ID' env if it's set.
    auto device_id_env = mindspore::common::GetEnv("DEVICE_ID");
    return device_id_env.empty() ? "0" : device_id_env;
  } else {
    // If multiple device number is set, we need to find the smallest device id. Set to "0" for now.
    return "0";
  }
}

static std::map<std::string, std::string> GenHcclOptions(uint32_t device_id, std::string_view rank_id,
                                                         std::string_view rank_file = "") {
  std::map<std::string, std::string> default_options_map = {
    {ge::OPTION_EXEC_IS_USEHCOM, "1"},         {ge::OPTION_EXEC_IS_USEHVD, "0"},
    {ge::OPTION_EXEC_HCCL_FLAG, "1"},          {ge::OPTION_EXEC_DEVICE_ID, std::to_string(device_id)},
    {ge::OPTION_EXEC_RANK_ID, rank_id.data()}, {ge::OPTION_EXEC_POD_NAME, rank_id.data()},
    {ge::OPTION_GRAPH_RUN_MODE, "1"},          {ge::OPTION_EXEC_HCCL_FLAG, "1"},
    {ge::OPTION_EXEC_DEPLOY_MODE, "0"}};

  if (!rank_file.empty()) {
    default_options_map.emplace(ge::OPTION_EXEC_RANK_TABLE_FILE, rank_file.data());
  }

  return default_options_map;
}

namespace mindspore::hccl {
namespace {
const char kDefaultGroup[] = "__default_group";
constexpr uint32_t kDeviceNumOfServer = 8;

std::string MapToJson(const std::map<std::string, std::string> &map) {
  std::stringstream ss;
  ss << "{";
  bool first = true;
  for (const auto &pair : map) {
    if (!first) {
      ss << ",";
    }
    ss << "\\\"" << pair.first << "\\\""
       << ": "
       << "\\\"" << pair.second << "\\\"";
    first = false;
  }
  ss << "}";
  return ss.str();
}

std::string GetMstxMsg(const std::string &opName, HcclDataType dataType, size_t dataCnt, HcclComm comm, size_t streamId,
                       uint32_t srcRank, uint32_t destRank) {
  static const std::map<HcclDataType, std::string> dataTypes = {
    {HCCL_DATA_TYPE_INT8, "int8"},     {HCCL_DATA_TYPE_INT16, "int16"},     {HCCL_DATA_TYPE_INT32, "int32"},
    {HCCL_DATA_TYPE_FP16, "fp16"},     {HCCL_DATA_TYPE_FP32, "fp32"},       {HCCL_DATA_TYPE_INT64, "int64"},
    {HCCL_DATA_TYPE_UINT64, "uint64"}, {HCCL_DATA_TYPE_UINT8, "uint8"},     {HCCL_DATA_TYPE_UINT16, "uint16"},
    {HCCL_DATA_TYPE_UINT32, "uint32"}, {HCCL_DATA_TYPE_FP64, "fp64"},       {HCCL_DATA_TYPE_BFP16, "bfp16"},
#ifdef EXPERIMENT_A5
    {HCCL_DATA_TYPE_HIF8, "hif8"},     {HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"}, {HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
#endif
  };
  constexpr int64_t MAX_GROUP_NAME_LEN = 128;
  static std::map<HcclComm, std::string> commNames;
  std::map<std::string, std::string> opDescMap;
  opDescMap["opName"] = opName;
  auto nameIter = commNames.find(comm);
  if (nameIter != commNames.end()) {
    opDescMap["groupName"] = nameIter->second;
  } else {
    opDescMap["groupName"] = "na";
    static char commName[MAX_GROUP_NAME_LEN] = {0};
    if (HcclGetCommName(comm, commName) == HCCL_SUCCESS) {
      std::string name(commName);
      opDescMap["groupName"] = name;
      commNames.insert({comm, name});
    }
  }
  if (srcRank != UINT32_MAX) {
    opDescMap["srcRank"] = std::to_string(srcRank);
  }
  if (destRank != UINT32_MAX) {
    opDescMap["destRank"] = std::to_string(destRank);
  }
  opDescMap["dataType"] = "na";
  auto iter = dataTypes.find(dataType);
  if (iter != dataTypes.end()) {
    opDescMap["dataType"] = iter->second;
  }
  opDescMap["count"] = std::to_string(dataCnt);
  opDescMap["streamId"] = std::to_string(streamId);
  std::string opDescMsg = MapToJson(opDescMap);
  return opDescMsg;
}
}  // namespace

HcclAdapter &HcclAdapter::GetInstance() {
  static HcclAdapter instance;
  return instance;
}

void HcclAdapter::InitPlugin() {
  if (plugin_handle_ != nullptr) {
    return;
  }
#ifndef ENABLE_ASAN
  plugin_handle_ = dlopen(kHcclPluginFileName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
#else
  plugin_handle_ = dlopen(kHcclPluginFileName, RTLD_NOW | RTLD_LOCAL);
#endif
  if (plugin_handle_ == nullptr) {
    MS_LOG(EXCEPTION) << "Dlopen " << kHcclPluginFileName << " failed, result = " << GetDlErrorMsg();
  }
  init_hcom_graph_adapter_ = DlsymFuncObj(InitHcomGraphAdapter, plugin_handle_);
  finalize_hcom_graph_adapter_ = DlsymFuncObj(FinalizeHcomGraphAdapter, plugin_handle_);
  get_hccl_comm_config_capability_ = DlsymFuncObj(HcclGetCommConfigCapability, plugin_handle_);
  get_hccl_kernel_info_store_ = DlsymFuncObj(GetHcclKernelInfoStore, plugin_handle_);
  get_all_kernel_builder_ = DlsymFuncObj(GetAllKernelBuilder, plugin_handle_);
  init_hccl_comm_ = DlsymFuncObj(HcclCommInitClusterInfo, plugin_handle_);
  finalize_hccl_comm_ = DlsymFuncObj(HcclCommDestroy, plugin_handle_);
  single_op_hccl_get_rank_id_ = DlsymFuncObj(HcclGetRankId, plugin_handle_);
  single_op_hccl_get_rank_size_ = DlsymFuncObj(HcclGetRankSize, plugin_handle_);
  hccl_get_comm_async_error_ = DlsymFuncObj(HcclGetCommAsyncError, plugin_handle_);
  hccl_get_error_string_ = DlsymFuncObj(HcclGetErrorString, plugin_handle_);
  launch_hccl_broadcast_ = DlsymFuncObj(HcclBroadcast, plugin_handle_);
  launch_hccl_all_reduce_ = DlsymFuncObj(HcclAllReduce, plugin_handle_);
  launch_hccl_reduce_ = DlsymFuncObj(HcclReduce, plugin_handle_);
  launch_hccl_scatter_ = DlsymFuncObj(HcclScatter, plugin_handle_);
  launch_hccl_reduce_scatter_ = DlsymFuncObj(HcclReduceScatter, plugin_handle_);
  launch_hccl_all_gather_ = DlsymFuncObj(HcclAllGather, plugin_handle_);
  launch_hccl_send_ = DlsymFuncObj(HcclSend, plugin_handle_);
  launch_hccl_recv_ = DlsymFuncObj(HcclRecv, plugin_handle_);
  launch_hccl_barrier_ = DlsymFuncObj(HcclBarrier, plugin_handle_);
  launch_hccl_batch_isend_irecv_ = DlsymFuncObj(HcclBatchSendRecv, plugin_handle_);
  hccl_create_group_ = DlsymFuncObj(HcomCreateGroup, plugin_handle_);
  hccl_destroy_group_ = DlsymFuncObj(HcomDestroyGroup, plugin_handle_);
  hccl_get_rank_id_ = DlsymFuncObj(HcomGetRankId, plugin_handle_);
  hccl_get_rank_size_ = DlsymFuncObj(HcomGetRankSize, plugin_handle_);
  hccl_get_local_rank_id_ = DlsymFuncObj(HcomGetLocalRankId, plugin_handle_);
  hccl_get_local_rank_size_ = DlsymFuncObj(HcomGetLocalRankSize, plugin_handle_);
  hccl_get_world_rank_by_group_rank_ = DlsymFuncObj(HcomGetWorldRankFromGroupRank, plugin_handle_);
  hccl_get_group_rank_by_world_rank_ = DlsymFuncObj(HcomGetGroupRankFromWorldRank, plugin_handle_);
  hccl_exec_initialize_ = DlsymFuncObj(HcomExecInitialize, plugin_handle_);
  hccl_exec_finalize_ = DlsymFuncObj(HcomExecFinalize, plugin_handle_);
  hccl_exec_enqueue_op_ = DlsymFuncObj(HcomExecEnqueueOperation, plugin_handle_);
  hccl_exec_enqueue_all_to_all_v_ = DlsymFuncObj(HcomExecEnqueueAllToAllV, plugin_handle_);
  launch_hccl_all_to_allv_ = DlsymFuncObj(HcclAlltoAllV, plugin_handle_);
  launch_hccl_all_to_allvc_ = DlsymAscendFuncObj(HcclAlltoAllVC, plugin_handle_);
  launch_hccl_reduce_scatterv_ = DlsymAscendFuncObj(HcclReduceScatterV, plugin_handle_);
  launch_hccl_all_gatherv_ = DlsymAscendFuncObj(HcclAllGatherV, plugin_handle_);
  launch_hccl_all_to_all_ = DlsymFuncObj(HcclAlltoAll, plugin_handle_);
  launch_hccl_comm_resume_ = DlsymAscendFuncObj(HcclCommResume, plugin_handle_);
  hcom_destroy_ = DlsymFuncObj(HcomDestroy, plugin_handle_);
}

void HcclAdapter::FinalizePlugin() {
  if (plugin_handle_ == nullptr) {
    return;
  }
  init_hcom_graph_adapter_ = nullptr;
  init_hccl_root_info_config_ = nullptr;
  init_hccl_global_comm_ranktable_ = nullptr;
  init_hccl_sub_comm_ranktable_ = nullptr;
  finalize_hcom_graph_adapter_ = nullptr;
  get_hccl_comm_config_capability_ = nullptr;
  get_hccl_kernel_info_store_ = nullptr;
  get_all_kernel_builder_ = nullptr;
  init_hccl_comm_ = nullptr;
  finalize_hccl_comm_ = nullptr;
  launch_hccl_broadcast_ = nullptr;
  launch_hccl_all_reduce_ = nullptr;
  launch_hccl_reduce_ = nullptr;
  launch_hccl_scatter_ = nullptr;
  launch_hccl_reduce_scatter_ = nullptr;
  launch_hccl_all_gather_ = nullptr;
  launch_hccl_send_ = nullptr;
  launch_hccl_recv_ = nullptr;
  launch_hccl_barrier_ = nullptr;
  launch_hccl_batch_isend_irecv_ = nullptr;
  hccl_create_group_ = nullptr;
  hccl_destroy_group_ = nullptr;
  hccl_get_rank_id_ = nullptr;
  hccl_get_local_rank_id_ = nullptr;
  hccl_get_local_rank_size_ = nullptr;
  hccl_get_world_rank_by_group_rank_ = nullptr;
  hccl_get_group_rank_by_world_rank_ = nullptr;
  hccl_get_rank_size_ = nullptr;
  hccl_exec_initialize_ = nullptr;
  hccl_exec_finalize_ = nullptr;
  hccl_exec_enqueue_op_ = nullptr;
  hccl_exec_enqueue_all_to_all_v_ = nullptr;
  hccl_comm_working_dev_nic_set_ = nullptr;
  launch_hccl_all_to_allv_ = nullptr;
  launch_hccl_all_to_allvc_ = nullptr;
  launch_hccl_reduce_scatterv_ = nullptr;
  launch_hccl_all_gatherv_ = nullptr;
  launch_hccl_comm_resume_ = nullptr;
  hcom_destroy_ = nullptr;
  (void)dlclose(plugin_handle_);
  plugin_handle_ = nullptr;
}

std::string HcclAdapter::GetHcclModeString(HcclMode hccl_mode) {
  static std::map<HcclMode, std::string> kHcclModeString = {{HcclMode::kGraph, "GE_MODE"},
                                                            {HcclMode::kPynative, "PYNATIVE_MODE"},
                                                            {HcclMode::kKernelByKernel, "KERNEL_BY_KERNEL_MODE"}};
  return kHcclModeString.at(hccl_mode);
}

bool HcclAdapter::InitHccl(uint32_t device_id, std::string_view rank_id) {
  MS_LOG(INFO) << "Start init hccl adapter.";
  common::SetEnv("HCCL_WHITELIST_DISABLE", "1");
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (init_flag_) {
    MS_LOG(INFO) << "Hccl has been inited, skip.";
    return true;
  }
  InitPlugin();
  bool ret = InitHcclExec();
  if (!ret) {
    return false;
  }
  init_flag_ = true;
  MS_LOG(INFO) << "Init hccl adapter success.";
  return true;
}

bool HcclAdapter::InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file,
                           HcclMode hccl_mode) {
  MS_LOG(INFO) << "Start init hccl adapter for " << GetHcclModeString(hccl_mode);
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (init_flag_) {
    MS_LOG(INFO) << "Hccl has been inited, skip.";
    return true;
  }

  hccl_mode_ = hccl_mode;
  InitPlugin();
  if (hccl_mode_ == HcclMode::kGraph) {
    auto options = GenHcclOptions(device_id, rank_id, rank_file);
    bool ret = InitKernelInfoStore(options);
    if (!ret) {
      return false;
    }

    ret = InitHcclExec();
    if (!ret) {
      return false;
    }
  } else {
    bool ret = InitHcclComm(rank_id, rank_file);
    if (!ret) {
      return false;
    }
  }

  init_flag_ = true;
  MS_LOG(INFO) << "Init hccl adapter success.";
  return true;
}

bool HcclAdapter::IsSimulation() {
  static bool dry_run = common::IsCompileSimulation() || common::IsExecuteSimulation();
  return dry_run;
}

bool HcclAdapter::HcclWatchdogThread(HcclComm comm, std::string *error_info, bool *disable) {
  if (!init_flag_ || hccl_get_comm_async_error_ == nullptr || hccl_get_error_string_ == nullptr) {
    MS_LOG(INFO) << "Hccl has never been inited, skip.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(disable);
  HcclResult hccl_async_error;
  auto ret = hccl_get_comm_async_error_(comm, &hccl_async_error);
  if (ret != HCCL_SUCCESS) {
    MS_LOG(WARNING) << "Call HcclGetCommAsyncError failed, close watchdog.";
    *disable = true;
    return true;
  }
  MS_LOG(DEBUG) << "hccl_get_comm_async_error_ res: " << hccl_async_error << ", comm: " << comm;
  if (hccl_async_error != HCCL_SUCCESS) {
    std::ostringstream oss;
    // handle special error code, this code is due to a network error or a remote process exiting prematurely.
    string e_remote_err;
    if (hccl_async_error == HCCL_E_REMOTE) {
      e_remote_err = "[ HCCL_E_REMOTE ] ";
    }
    oss << "Hccl get comm async error failed, error code is: " << hccl_async_error << ", detail info: " << e_remote_err
        << hccl_get_error_string_(hccl_async_error);
    *error_info = oss.str();
    return false;
  }
  return true;
}

bool HcclAdapter::FinalizeHccl() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  MS_LOG(INFO) << "Start destroy hccl adapter for " << GetHcclModeString(hccl_mode_);
  if (!init_flag_) {
    MS_LOG(INFO) << "Hccl has never been inited, skip.";
    return true;
  }
  (void)FinalizeHcclExec();
  (void)FinalizeKernelInfoStore();
  (void)FinalizeHcclComm();
  if (hcom_destroy_ != nullptr) {
    hcom_destroy_();
  }
  FinalizePlugin();
  init_flag_ = false;
  MS_LOG(INFO) << "Destroy hccl adapter success.";
  return true;
}

HcclResult HcclAdapter::HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                      aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_broadcast_(buf, count, dataType, root, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                      const HcclReduceOp op, const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_all_reduce_(send_buf, recv_buf, count, dataType, op, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                   HcclReduceOp op, uint32_t root, const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_reduce_(send_buf, recv_buf, count, dataType, op, root, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                    uint32_t root, HcclComm comm, aclrtStream stream) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_scatter_(send_buf, recv_buf, count, dataType, root, comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclReduceScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                          const HcclReduceOp op, const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_reduce_scatter_(send_buf, recv_buf, count, dataType, op, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclAllGather(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                      const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_all_gather_(send_buf, recv_buf, count, dataType, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclSend(void *send_buf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                 const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, -1, destRank).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_send_(send_buf, count, dataType, destRank, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclRecv(void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                 const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, count, hccl_comm, streamId, srcRank, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_recv_(recv_buf, count, dataType, srcRank, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclBarrier(const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  return launch_hccl_barrier_(hccl_comm, stream);
}

HcclResult HcclAdapter::HcclBatchISendIRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum, HcclComm comm,
                                            aclrtStream stream) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId,
               GetMstxMsg(__func__, sendRecvInfo[0].dataType, sendRecvInfo[0].count, comm, streamId, -1, -1).c_str(),
               stream, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_batch_isend_irecv_(sendRecvInfo, itemNum, comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclCommResume(HcclComm comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  if (launch_hccl_comm_resume_ == nullptr) {
    MS_LOG(EXCEPTION) << "Dynamically load HcclCommResume failed.";
  }
  return launch_hccl_comm_resume_(comm);
}

bool HcclAdapter::InitKernelInfoStore(const std::map<std::string, std::string> options) {
  MS_LOG(INFO) << "Start init hccl kernel info store.";
  MS_EXCEPTION_IF_NULL(init_hcom_graph_adapter_);
  MS_EXCEPTION_IF_NULL(get_hccl_kernel_info_store_);
  // get ops_kernel_builder
  std::map<std::string, std::shared_ptr<ge::OpsKernelBuilder>> all_builders;
  get_all_kernel_builder_(&all_builders);
  auto iter = all_builders.find(kHcclOpsKernelInfoStore);
  if (iter == all_builders.end()) {
    std::string all_builders_name = "[";
    for (const auto &it : all_builders) {
      all_builders_name += it.first + " ";
    }
    all_builders_name += "]";
    MS_LOG(EXCEPTION) << "Builders size " << all_builders.size() << ", cannot find " << kHcclOpsKernelInfoStore
                      << ", full list of builders: " << all_builders_name;
  }

  MS_LOG(INFO) << "Get builder " << iter->first;
  ops_kernel_builder_ = iter->second;
  MS_EXCEPTION_IF_NULL(ops_kernel_builder_);
  // init ops_kernel_builder
  auto ret = ops_kernel_builder_->Initialize(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init hccl kernel builder failed.";
  }

  // get ops_kernel_info_store
  ret = init_hcom_graph_adapter_(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init hccl graph adapter failed.";
  }

  get_hccl_kernel_info_store_(&ops_kernel_info_store_);
  MS_EXCEPTION_IF_NULL(ops_kernel_info_store_);
  ret = ops_kernel_info_store_->Initialize(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init info store failed.";
  }
  init_kernel_info_store_ = true;
  MS_LOG(INFO) << "Init hccl kernel info store success.";
  return true;
}

bool HcclAdapter::FinalizeKernelInfoStore() {
  if (!init_kernel_info_store_) {
    return true;
  }
  MS_LOG(INFO) << "Start destroy hccl kernel info store.";
  if (ops_kernel_info_store_ != nullptr) {
    auto ret = ops_kernel_info_store_->Finalize();
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Destroy info store failed, ret = " << ret;
      return false;
    }
  }

  if (ops_kernel_builder_ != nullptr) {
    auto ret = ops_kernel_builder_->Finalize();
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Destroy builder failed, ret = " << ret;
      return false;
    }
  }

  MS_EXCEPTION_IF_NULL(finalize_hcom_graph_adapter_);
  finalize_hcom_graph_adapter_();
  ops_kernel_info_store_.reset();
  ops_kernel_builder_.reset();
  init_kernel_info_store_ = false;
  MS_LOG(INFO) << "Destroy hccl kernel info store success.";
  return true;
}

uint32_t HcclAdapter::HcclGetCommConfigCapability() {
  MS_EXCEPTION_IF_NULL(get_hccl_comm_config_capability_);
  return get_hccl_comm_config_capability_();
}

HcclResult HcclAdapter::HcclCommInitClusterInfoConfig(const char *rank_table, uint32_t rank_id, HcclCommConfig *config,
                                                      HcclComm *hccl_comm) {
  if (init_hccl_global_comm_ranktable_ == nullptr) {
    init_hccl_global_comm_ranktable_ = DlsymFuncObj(HcclCommInitClusterInfoConfig, plugin_handle_);
  }
  return init_hccl_global_comm_ranktable_(rank_table, rank_id, config, hccl_comm);
}

HcclResult HcclAdapter::HcclCommInitRootInfoConfig(uint32_t n_ranks, const HcclRootInfo *root_info, uint32_t rank,
                                                   const HcclCommConfig *config, HcclComm *hccl_comm_) {
  if (init_hccl_root_info_config_ == nullptr) {
    init_hccl_root_info_config_ = DlsymFuncObj(HcclCommInitRootInfoConfig, plugin_handle_);
    if (init_hccl_root_info_config_ == nullptr) {
      // new api in CANN C20
      return HcclCommInitRootInfo(n_ranks, root_info, rank, hccl_comm_);
    }
  }

  return init_hccl_root_info_config_(n_ranks, root_info, rank, config, hccl_comm_);
}

HcclResult HcclAdapter::HcclCreateSubCommConfig(HcclComm *global_comm, uint32_t rank_size, uint32_t *rank_ids,
                                                uint64_t comm_id, uint32_t rank_id, HcclCommConfig *config,
                                                HcclComm *hccl_comm) {
  if (init_hccl_sub_comm_ranktable_ == nullptr) {
    init_hccl_sub_comm_ranktable_ = DlsymFuncObj(HcclCreateSubCommConfig, plugin_handle_);
  }
  return init_hccl_sub_comm_ranktable_(global_comm, rank_size, rank_ids, comm_id, rank_id, config, hccl_comm);
}

bool HcclAdapter::InitHcclComm(std::string_view rank_id, std::string_view rank_file) {
  MS_LOG(INFO) << "Start init hccl comm.";
  int rank_id_i = -1;
  try {
    rank_id_i = std::stoi(rank_id.data());
  } catch (std::invalid_argument &) {
    MS_LOG(EXCEPTION) << "Invalid rank id env:" << rank_id;
  }
  if (rank_id_i < 0) {
    MS_LOG(ERROR) << "rank_id cannot be negative";
    return false;
  }
  MS_EXCEPTION_IF_NULL(init_hccl_comm_);
  auto hccl_result = init_hccl_comm_(rank_file.data(), rank_id_i, &hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclCommInitClusterInfo failed, ret:" << hccl_result;
    return false;
  }
  MS_LOG(INFO) << "InitHcclComm success";
  return true;
}

bool HcclAdapter::FinalizeHcclComm() {
  MS_LOG(INFO) << "Start finalize hccl comm.";
  if (hccl_comm_ == nullptr) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(finalize_hccl_comm_);
  auto hccl_result = finalize_hccl_comm_(hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclComm destroy failed, ret:" << hccl_result;
    return false;
  }
  hccl_comm_ = nullptr;
  MS_LOG(INFO) << "HcclComm destroy success";
  return true;
}

HcclResult HcclAdapter::HcclCreateGroup(const std::string &group, uint32_t rank_num, uint32_t *rank_ids) const {
  CHECK_SYMBOL_NULL(hccl_create_group_);
  return hccl_create_group_(group.c_str(), rank_num, rank_ids);
}

HcclResult HcclAdapter::HcclDestroyGroup(const std::string &group) const {
  CHECK_SYMBOL_NULL(hccl_destroy_group_);
  return hccl_destroy_group_(group.c_str());
}

HcclResult HcclAdapter::HcclGetRankId(const std::string &group, uint32_t *rank_id) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(single_op_hccl_get_rank_id_);
    return single_op_hccl_get_rank_id_(hccl_comm_, rank_id);
  } else {
    CHECK_SYMBOL_NULL(hccl_get_rank_id_);
    return hccl_get_rank_id_(group.c_str(), rank_id);
  }
}

HcclResult HcclAdapter::HcclGetRankSize(const std::string &group, uint32_t *rank_size) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(single_op_hccl_get_rank_size_);
    return single_op_hccl_get_rank_size_(hccl_comm_, rank_size);
  } else {
    CHECK_SYMBOL_NULL(hccl_get_rank_size_);
    return hccl_get_rank_size_(group.c_str(), rank_size);
  }
}

HcclResult HcclAdapter::HcclGetLocalRankId(const std::string &group, uint32_t *local_rank_id) const {
  CHECK_SYMBOL_NULL(hccl_get_local_rank_id_);
  return hccl_get_local_rank_id_(group.c_str(), local_rank_id);
}

HcclResult HcclAdapter::HcclGetLocalRankSize(const std::string &group, uint32_t *local_rank_size) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    MS_LOG(ERROR) << "The pynative mode doesn't support get local rank szie.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_local_rank_size_);
    return hccl_get_local_rank_size_(group.c_str(), local_rank_size);
  }
}

HcclResult HcclAdapter::HcclGetWorldRankFromGroupRank(const std::string &group, uint32_t local_rank,
                                                      uint32_t *world_rank) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    MS_LOG(ERROR) << "The pynative mode doesn't support get world rank by group rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_world_rank_by_group_rank_);
    return hccl_get_world_rank_by_group_rank_(group.c_str(), local_rank, world_rank);
  }
}

HcclResult HcclAdapter::HcclGetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group,
                                                      uint32_t *local_rank) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    MS_LOG(ERROR) << "The pynative mode doesn't support get group rank by world rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_group_rank_by_world_rank_);
    return hccl_get_group_rank_by_world_rank_(world_rank, group.c_str(), local_rank);
  }
}

HcclResult HcclAdapter::HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks) {
  if (hccl_comm_working_dev_nic_set_ == nullptr) {
    hccl_comm_working_dev_nic_set_ = DlsymFuncObj(HcclCommWorkingDevNicSet, plugin_handle_);
  }
  CHECK_SYMBOL_NULL(hccl_comm_working_dev_nic_set_);
  return hccl_comm_working_dev_nic_set_(comm, ranks, useBackup, nRanks);
}

bool HcclAdapter::InitHcclExec() {
  MS_LOG(INFO) << "Start init hccl exec.";
  MS_EXCEPTION_IF_NULL(hccl_exec_initialize_);
  HcclResult hccl_ret = hccl_exec_initialize_();
  if (hccl_ret == HCCL_E_PTR) {
    MS_LOG(WARNING) << "Hccl comm is null, hcom executor initialize is not required";
  } else if (hccl_ret == HCCL_SUCCESS) {
    MS_LOG(INFO) << "Hcom DynamicKernel Initialize success";
  } else {
    MS_LOG(ERROR) << "Hcom DynamicKernel Initialize failed";
    return false;
  }
  init_hccl_exec_ = true;
  MS_LOG(INFO) << "InitHcclExec success";
  return true;
}

bool HcclAdapter::FinalizeHcclExec() {
  if (!init_hccl_exec_) {
    return true;
  }
  MS_LOG(INFO) << "Start finalize hccl exec.";
  MS_EXCEPTION_IF_NULL(hccl_exec_finalize_);
  HcclResult hccl_ret = hccl_exec_finalize_();
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Hcom DynamicKernel Finalize failed";
    return false;
  }
  init_hccl_exec_ = false;
  MS_LOG(INFO) << "HcclExec destroy success";
  return true;
}

HcclResult HcclAdapter::HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const {
  CHECK_SYMBOL_NULL(hccl_exec_enqueue_op_);
  return hccl_exec_enqueue_op_(op_info, callback);
}

HcclResult HcclAdapter::HcclExecAlltoAllV(const ::HcomAllToAllVParams &params, const HExecCallBack &callback) const {
  CHECK_SYMBOL_NULL(hccl_exec_enqueue_all_to_all_v_);
  return hccl_exec_enqueue_all_to_all_v_(params, callback);
}

bool HcclAdapter::UseHcclCM() const {
  // This MS_HCCL_CM_INIT env is deperacated since MindSpore 2.3 version.
  return false;
}

HcclResult HcclAdapter::HcclAlltoAllV(void *send_buf, void *recv_buf, hccl::HcclAllToAllVParams params,
                                      HcclDataType dataType, aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  CHECK_SYMBOL_NULL(launch_hccl_all_to_allv_);
  MS_EXCEPTION_IF_NULL(hccl_comm);
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    uint64_t counts = static_cast<uint64_t>(params.sendcounts.size());
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, counts, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret =
    launch_hccl_all_to_allv_(send_buf, params.sendcounts.data(), params.sdispls.data(), dataType, recv_buf,
                             params.recvcounts.data(), params.rdispls.data(), dataType, hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

HcclResult HcclAdapter::HcclAlltoAllVC(void *send_buf, void *send_count_matrix, HcclDataType send_type, void *recv_buf,
                                       HcclDataType recv_type, aclrtStream stream, HcclComm hccl_comm) const {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return HCCL_SUCCESS;
  }
  CHECK_SYMBOL_NULL(launch_hccl_all_to_allvc_);
  MS_EXCEPTION_IF_NULL(hccl_comm);
  HcclResult ret =
    launch_hccl_all_to_allvc_(send_buf, send_count_matrix, send_type, recv_buf, recv_type, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclReduceScatterV(void *send_buf, void *recv_buf, hccl::HcclReduceScatterVParams params,
                                           HcclDataType data_type, const HcclReduceOp op, const aclrtStream stream,
                                           HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  CHECK_SYMBOL_NULL(launch_hccl_reduce_scatterv_);
  MS_EXCEPTION_IF_NULL(hccl_comm);
  HcclResult ret = launch_hccl_reduce_scatterv_(send_buf, params.send_counts.data(), params.sdispls.data(), recv_buf,
                                                params.recv_count, data_type, op, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllGatherV(void *send_buf, void *recv_buf, hccl::HcclAllGatherVParams params,
                                       HcclDataType data_type, const aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  CHECK_SYMBOL_NULL(launch_hccl_all_gatherv_);
  MS_EXCEPTION_IF_NULL(hccl_comm);
  HcclResult ret = launch_hccl_all_gatherv_(send_buf, params.send_count, recv_buf, params.recv_counts.data(),
                                            params.rdispls.data(), data_type, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllToAll(void *send_buf, void *recv_buf, hccl::HcclAllToAllParams params,
                                     HcclDataType dataType, aclrtStream stream, HcclComm hccl_comm) const {
  if (MS_UNLIKELY(IsSimulation())) {
    return HCCL_SUCCESS;
  }
  CHECK_SYMBOL_NULL(launch_hccl_all_to_all_);
  MS_EXCEPTION_IF_NULL(hccl_comm);
  uint64_t rangeId = 0;
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    size_t streamId = mindspore::device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    MSTX_START(rangeId, GetMstxMsg(__func__, dataType, params.sendcount, hccl_comm, streamId, -1, -1).c_str(), stream,
               mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  HcclResult ret = launch_hccl_all_to_all_(send_buf, params.sendcount, dataType, recv_buf, params.recvcount, dataType,
                                           hccl_comm, stream);
  if (MS_UNLIKELY(mindspore::profiler::MstxImpl::GetInstance().IsEnable())) {
    MSTX_END(rangeId, mindspore::profiler::MSTX_DOMAIN_COMMUNICATION);
  }
  return ret;
}

bool HcclAdapter::IsSameServer(const std::vector<uint32_t> &rank_ids) const {
  auto min_iter = min_element(rank_ids.begin(), rank_ids.end());
  uint32_t min = (min_iter != rank_ids.end()) ? *min_iter : 0;
  auto max_iter = max_element(rank_ids.begin(), rank_ids.end());
  uint32_t max = (max_iter != rank_ids.end()) ? *max_iter : 0;
  return ((max - min < kDeviceNumOfServer) && (min / kDeviceNumOfServer == max / kDeviceNumOfServer));
}

string HcclAdapter::GetHcomGroup(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  auto rank_ids = common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, cnode)
                    ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrGroupRankIds)
                    : std::vector<uint32_t>();
  auto new_group = DoGetHcomGroup(group_name, rank_ids);

  MS_LOG(INFO) << "hcom node: " << cnode->fullname_with_scope() << ", old group: " << group_name
               << ", new group: " << new_group;

  if (cnode->HasAttr(parallel::FIRST_RECEIVE)) {
    return new_group;
  }
  const auto &node_name = common::AnfAlgo::GetCNodeName(cnode);
  static const auto send_recv_parallel = (common::GetEnv("SEND_RECV_PARALLEL") == "1");
  if ((node_name == kSendOpName || node_name == kReceiveOpName) && !send_recv_parallel) {
    MS_LOG(DEBUG) << "hcom node: " << cnode->fullname_with_scope() << "is set to group: -1.";
    return "-1";
  }

  return new_group;
}

string HcclAdapter::DoGetHcomGroup(const string &original_group, const std::vector<uint32_t> &rank_ids) const {
  string communi_parallel_mode = parallel::ParallelContext::GetInstance()->communi_parallel_mode();
  if (communi_parallel_mode == parallel::kAllGroupParallel) {
    return original_group;
  }

  if (communi_parallel_mode == parallel::kNoGroupParallel) {
    return kDefaultGroup;
  }

  if (rank_ids.empty() || original_group == kHcclWorldGroup) {
    return kDefaultGroup;
  }

  if (IsSameServer(rank_ids)) {
    return original_group;
  }

  return kDefaultGroup;
}

void HcclAdapter::AddCMEnvToHcclOption(std::map<std::string, std::string> *hccl_opt_map) {
  MS_EXCEPTION_IF_NULL(hccl_opt_map);
  // Before hccl initialization, the validation of these environment variables is already verified.
  hccl_opt_map->emplace(ge::OPTION_EXEC_CM_CHIEF_IP,
                        mindspore::common::GetEnv(mindspore::distributed::kEnvSchedulerHost));

  // We offset scheduler port by 1 as cm chief port.
  std::string sched_port = mindspore::common::GetEnv(mindspore::distributed::kEnvSchedulerPort);
  int sched_port_num = std::stoi(sched_port);
  int cm_port_num = sched_port_num + 1;
  hccl_opt_map->emplace(ge::OPTION_EXEC_CM_CHIEF_PORT, std::to_string(cm_port_num));
  hccl_opt_map->emplace(ge::OPTION_EXEC_CM_CHIEF_DEVICE, GenerateCMChiefWorkDevice());
  hccl_opt_map->emplace(ge::OPTION_EXEC_CM_WORKER_SIZE,
                        mindspore::common::GetEnv(mindspore::distributed::kEnvWorkerNum));
  hccl_opt_map->emplace(ge::OPTION_EXEC_CM_WORKER_IP, mindspore::common::GetEnv(mindspore::distributed::kEnvWorkerIp));
  MS_LOG(INFO) << "Set CM options to hccl. OPTION_EXEC_CM_CHIEF_IP: " << hccl_opt_map->at(ge::OPTION_EXEC_CM_CHIEF_IP)
               << ", OPTION_EXEC_CM_CHIEF_PORT: " << hccl_opt_map->at(ge::OPTION_EXEC_CM_CHIEF_PORT)
               << ", OPTION_EXEC_CM_CHIEF_DEVICE: " << hccl_opt_map->at(ge::OPTION_EXEC_CM_CHIEF_DEVICE)
               << ", OPTION_EXEC_CM_WORKER_SIZE: " << hccl_opt_map->at(ge::OPTION_EXEC_CM_WORKER_SIZE)
               << ", OPTION_EXEC_CM_WORKER_IP: " << hccl_opt_map->at(ge::OPTION_EXEC_CM_WORKER_IP);
}
}  // namespace mindspore::hccl
