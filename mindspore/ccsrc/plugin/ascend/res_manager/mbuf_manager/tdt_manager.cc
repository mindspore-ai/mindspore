/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/mbuf_manager/tdt_manager.h"
#include <algorithm>
#include <tuple>
#include <utility>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "utils/log_adapter.h"
#include "include/utils/callback.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorprint_utils.h"
#include "plugin/ascend/profiler/parallel_strategy_profiling.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorsummary_utils.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "plugin/ascend/res_manager/mbuf_manager/mbuf_receive_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
std::mutex g_tsd_mutex;
constexpr auto kPrintOpName = "Print";
constexpr auto kTensorDumpOpName = "TensorDump";
constexpr auto kTensorDumpChannelName = "ms_tensor_dump";
using MbufDataItem = std::variant<std::string, mindspore::tensor::TensorPtr>;
}  // namespace

TdtManager &TdtManager::GetInstance() {
  static TdtManager instance;
  return instance;
}

bool TdtManager::OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return true;
  }

  if (UseSimulationApi()) {
    return true;
  }

  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) != 0) {
    MS_LOG(DEBUG) << "ACLTDT Dataset client is already opened.";
    ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
    return true;
  }

  auto role = common::GetEnv("MS_ROLE");
  if (strcmp(role.c_str(), "MS_SCHED") == 0 || strcmp(role.c_str(), "MS_PSERVER") == 0) {
    return true;
  }

  uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  uint32_t rank_size;
  auto rank_size_env = common::GetEnv("RANK_SIZE");
  if (rank_size_env.empty()) {
    MS_LOG(INFO) << "Should config rank size.";
    rank_size = 1;
  } else {
    int rank_env = std::stoi(rank_size_env);
    if (rank_env <= 0) {
      MS_LOG(EXCEPTION) << "Error rank size " << rank_env << ".";
    }
    rank_size = IntToUint(rank_env);
  }
  (void)ErrorManagerAdapter::Init();
  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";
  auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id));
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret[" << static_cast<int>(ret)
                      << "]. The details refer to 'Ascend Error Message'.";
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);

  if (!ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    MbufDataHandlerManager::GetInstance().AddHandler(
      std::make_unique<MbufDataHandler>(std::bind(&TensorPrintUtils::PrintReceiveData, &TensorPrintUtils::GetInstance(),
                                                  std::placeholders::_1, std::placeholders::_2),
                                        device_id, kChannelNameNpuLog, kPrintOpName));
  }

  if (ms_context_ptr->backend_policy() == "ge") {
    constexpr char kMbufTensorDumpCallback[] = "MbufTensorDumpCallback";
    static auto tensordump_callback =
      callback::CommonCallback::GetInstance().GetCallback<void, const std::string &, const std::vector<MbufDataItem> &>(
        kMbufTensorDumpCallback);
    if (tensordump_callback) {
      device::ascend::MbufDataHandlerManager::GetInstance().AddHandler(
        std::make_unique<device::ascend::MbufDataHandler>(tensordump_callback, device_id, kTensorDumpChannelName,
                                                          kTensorDumpOpName));
    } else {
      MS_LOG(WARNING) << "Failed to get MbufTensorDumpCallback, tensor dump function may not work.";
    }
    if (TensorReportUtils::IsEnable()) {
      MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<MbufDataHandler>(
        std::bind(&TensorReportUtils::ReportReceiveData, &TensorReportUtils::GetInstance(), std::placeholders::_1,
                  std::placeholders::_2),
        device_id, tensorreport_mapping.first, tensorreport_mapping.second));
    }
    for (const std::pair<string, string> &summary_mapping : summary_mappings) {
      MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<MbufDataHandler>(
        std::bind(SummaryReceiveData, std::placeholders::_1, std::placeholders::_2, summary_mapping.first), device_id,
        summary_mapping.first, summary_mapping.second));
    }
  }
  return true;
}

bool TdtManager::CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_LOG(INFO) << "Start to close tsd, ref = " << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF);
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_TSD_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_TSD_REF, 0);
    pybind11::gil_scoped_release gil_release;
    MbufDataHandlerManager::GetInstance().DestoryPrintHandler();
    MbufDataHandlerManager::GetInstance().DestoryHandler();
    (void)ErrorManagerAdapter::Init();
    uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = CALL_ASCEND_API(aclrtResetDevice, static_cast<int32_t>(device_id));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtResetDevice failed, ret[" << static_cast<int>(ret)
                        << "]. The details refer to 'Ascend Error Message'.";
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call aclrtResetDevice, destroy and close tsd successful, ret[" << static_cast<int>(ret) << "]";
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
