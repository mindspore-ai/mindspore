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
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include <dlfcn.h>
#include <libgen.h>
#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "include/utils/callback.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "primitive/image_op_name.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_exception.h"
#include "include/utils/python_adapter.h"

namespace mindspore::device::ascend {
namespace {
constexpr char kOptimizerEndFlag[] = "optimizer_end";
constexpr char kOptimizerSnapshotFlag[] = "snapshot";

std::string GetMindIOPath() {
  try {
    const char mindio_pkg_name[] = "mindio_ttp";
    py::gil_scoped_acquire acquire;
    py::module mindio_ttp = python_adapter::GetPyModule(mindio_pkg_name);
    py::list path_attr = python_adapter::GetPyObjAttr(mindio_ttp, "__path__");
    return py::cast<std::string>(path_attr[0]);
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Try to get mindio path failed: " << e.what();
    return "";
  }
}
}  // namespace

OptimizerEventInfo &OptimizerEventInfo::GetInstance() {
  static OptimizerEventInfo instance;
  return instance;
}

void OptimizerEventInfo::RecordEvent(bool is_optimizer_start, void *stream) {
  auto &opt_event = (is_optimizer_start ? optimizer_start_event_ : optimizer_end_event_);
  if (opt_event == nullptr) {
    if (aclrtCreateEventExWithFlag(&opt_event, ACL_EVENT_TIME_LINE) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Create event for uce " << (is_optimizer_start ? "start" : "end") << " timestamp failed.";
      return;
    } else {
      MS_LOG(INFO) << "Create event for uce" << (is_optimizer_start ? "start" : "end") << " timestamp successfully.";
    }
  }
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Call aclrtRecordEvent for optimizer " << (is_optimizer_start ? "start" : "end")
                                    << " opt_event=" << opt_event << ", addr_of_opt_event=" << &opt_event;
  (void)CALL_ASCEND_API(aclrtRecordEvent, opt_event, stream);
}

void OptimizerEventInfo::GetOptimizerTimestamp(bool is_optimizer_start) {
  auto &opt_event = (is_optimizer_start ? optimizer_start_event_ : optimizer_end_event_);
  uint64_t timestamp = 0;
  aclError ret_code = CALL_ASCEND_API(aclrtEventGetTimestamp, opt_event, &timestamp);
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Call aclrtEventGetTimestamp for optimizer "
                                    << (is_optimizer_start ? "start" : "end") << " ret_code=" << ret_code
                                    << ", timestamp=" << timestamp << ", opt_event=" << opt_event;
  if (ret_code == ACL_SUCCESS) {
    if (is_optimizer_start) {
      optimizer_start_timestamp_ = timestamp;
    } else {
      optimizer_end_timestamp_ = timestamp;
    }
  } else {
    MS_LOG(ERROR) << "Call aclrtEventGetTimestamp for optimizer " << (is_optimizer_start ? "start" : "end")
                  << " ret_code=" << ret_code << ".";
  }
}

OptStartType OptimizerEventInfo::GetOptimizerStartType(kernel::KernelMod *kernel_mod, const CNodePtr &kernel) {
  if (optimizer_start_kernel_mod_ != nullptr) {
    return optimizer_start_kernel_mod_ == kernel_mod ? opt_start_type_ : OptStartType::OPT_START_TYPE_NONE;
  }
  if (kernel_mod->kernel_name() != kTensorReport) {
    return OptStartType::OPT_START_TYPE_NONE;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(kOptimizerEndFlag)) {
    optimizer_start_kernel_mod_ = kernel_mod;
    opt_start_type_ = prim->HasAttr(kOptimizerSnapshotFlag) ? OptStartType::OPT_START_TYPE_SNAPSHOT
                                                            : OptStartType::OPT_START_TYPE_REPORT;
    return opt_start_type_;
  }
  return OptStartType::OPT_START_TYPE_NONE;
}

bool OptimizerEventInfo::IsOptimizerEndKernelMod(kernel::KernelMod *kernel_mod, const CNodePtr &kernel) {
  if (optimizer_end_kernel_mod_ != nullptr) {
    return optimizer_end_kernel_mod_ == kernel_mod;
  }
  if (kernel_mod->kernel_name() != kTensorReport) {
    return false;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(kOptimizerEndFlag)) {
    optimizer_end_kernel_mod_ = kernel_mod;
    return true;
  }
  return false;
}

TensorReportUtils &TensorReportUtils::GetInstance() {
  static TensorReportUtils instance;
  static std::once_flag instInitFlag = {};
  std::call_once(instInitFlag, [&]() {
    std::string mindio_path = GetMindIOPath();
    if (mindio_path.empty()) {
      MS_LOG(WARNING) << "MindIO feature is switched on, but can't find MindIO package";
      return;
    }

    // open dependent library libttp_framework.so for libttp_c_api.so
    const std::string ttp_framework_file = "/framework_ttp/libttp_framework.so";
    auto framework_so_path = mindio_path + ttp_framework_file;
    if (dlopen(framework_so_path.c_str(), RTLD_LAZY) == nullptr) {
      MS_LOG(WARNING) << "MindIO feature is switched on, but try to open library " << ttp_framework_file
                      << " failed, error: " << dlerror();
      return;
    }

    const std::string ttp_so_file = "/mindspore_api/libttp_c_api.so";
    auto ttp_so_path = mindio_path + ttp_so_file;
    void *handle = dlopen(ttp_so_path.c_str(), RTLD_LAZY);
    MS_LOG(DEBUG) << "Start dlopen TFT so file " << ttp_so_path << ".";
    if (handle == nullptr) {
      MS_LOG(WARNING) << "MindIO feature is switched on, but try to open library " << ttp_so_path
                      << " failed, error: " << dlerror();
      return;
    }
    MS_LOG(INFO) << "dlopen TFT so file " << ttp_so_path << " successful.";
    auto tft_update_start_func = DlsymWithCast<TFT_StartUpdatingOsFunPtr>(handle, "MindioTtpSetOptimStatusUpdating");
    if (tft_update_start_func) {
      MS_LOG(INFO) << "Found TFT optimizer status updating function.";
      instance.SetTFTCallBack(tft_update_start_func);
    } else {
      MS_LOG(WARNING) << "MindIO feature is switched on, but can't find report function: "
                         "MindioTtpSetOptimStatusUpdating; Please check if MindIO package installed correctly!";
    }
  });
  return instance;
}

bool TensorReportUtils::IsEnable() {
  static std::vector<std::string_view> options = {"UCE:1", "TTP:1", "ARF:1", "HCCE:1"};
  return std::any_of(options.begin(), options.end(), [](const std::string_view &opt) {
    static std::string tftEnv = common::GetEnv("MS_ENABLE_TFT");
    return tftEnv.find(opt) != std::string::npos;
  });
}

TensorReportUtils::TensorReportUtils() {}

TensorReportUtils::~TensorReportUtils() {}

void TensorReportUtils::ReportReceiveData(
  const std::string &tensor_name,
  const std::vector<std::variant<std::string, mindspore::tensor::TensorPtr>> &data_items) {
  MS_LOG(DEBUG) << "Enter report recevice data.";
  static auto enable_uce_cb = GET_COMMON_CALLBACK(IsEnableUce, bool);
  if (enable_uce_cb != nullptr && enable_uce_cb()) {
    OptimizerEventInfo::GetInstance().GetOptimizerTimestamp(true);
  }
  if (_optStart != nullptr) {
    auto ret = _optStart(-1);
    MS_LOG(INFO) << "Send start updating optimizer event to TFT. ret=" << ret;
  }
  MS_LOG(DEBUG) << "Finish report recevice data.";
}

void TensorReportUtils::SetTFTCallBack(const TFT_StartUpdatingOsFunObj &optStart) { _optStart = optStart; }
}  // namespace mindspore::device::ascend
