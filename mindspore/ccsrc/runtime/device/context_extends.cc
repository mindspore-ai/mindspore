/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/context_extends.h"
#include <cstdlib>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include "pybind11/pybind11.h"
#include "include/common/utils/config_manager.h"
#include "utils/ms_utils.h"
#include "utils/convert_utils_base.h"
#ifndef NO_DLIB
#include "acl/acl_tdt.h"
#include "runtime/dev.h"
#include "toolchain/plog.h"
#include "common/util/error_manager/error_manager.h"
#endif
#ifdef ENABLE_D
#include "debug/data_dump/dump_json_parser.h"
#include "include/transform/graph_ir/utils.h"
#endif
#include "profiler/device/profiling.h"

namespace py = pybind11;

namespace mindspore {
namespace context {
#ifdef ENABLE_D
namespace {
constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
const std::vector<std::string> kGeDumpMode = {"all", "input", "output"};
}  // namespace
#endif

constexpr auto kUnknowErrorString = "Unknown error occurred";
#ifndef NO_DLIB
// Open tdt dataset
bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }

  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
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

  uint32_t rank_size = 1;
  uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

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

  int log_ret = DlogReportInitialize();
  if (log_ret != 0) {
    MS_LOG(WARNING) << "Init slog failed, ret = " << log_ret;
  }

  if (ErrorManager::GetInstance().Init() != 0) {
    MS_LOG(WARNING) << "Init ascend error manager failed, some ascend error log may be left out.";
  }
  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";
  auto ret = rtSetDevice(static_cast<int32_t>(device_id));
  if (ret != RT_ERROR_NONE) {
    const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
    if (!error_message.empty() && error_message.find(kUnknowErrorString) == std::string::npos) {
      MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
    }
    MS_LOG(EXCEPTION) << "Device " << device_id << " call rtSetDevice failed, ret[" << static_cast<int>(ret) << "]";
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
#ifdef ENABLE_TDTQUE
  auto thread_crt = [](const std::string &path, const acltdtChannelHandle *acl_handle) {
    return std::thread(TensorPrint(path, acl_handle));
  };
  ms_context_ptr->CreateTensorPrintThread(thread_crt);
#endif
  return true;
}

bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context_prt is nullptr";
  }
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_TSD_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_TSD_REF, 0);

#ifdef ENABLE_TDTQUE
    py::gil_scoped_release gil_release;
    ms_context_ptr->DestroyTensorPrintThread();
#endif
    if (ErrorManager::GetInstance().Init() != 0) {
      MS_LOG(WARNING) << "Init ascend error manager failed, some ascend error log may be left out.";
    }
    uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = rtDeviceReset(static_cast<int32_t>(device_id));
    if (ret != RT_ERROR_NONE) {
      const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
      if (!error_message.empty() && error_message.find(kUnknowErrorString) == std::string::npos) {
        MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
      }
      MS_LOG(EXCEPTION) << "Device " << device_id << " call rtDeviceReset failed, ret[" << static_cast<int>(ret) << "]";
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call rtDeviceReset, destroy and close tsd successful, ret[" << static_cast<int>(ret) << "]";
    (void)DlogReportFinalize();
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}
#else
bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) { return true; }
bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool) { return true; }
#endif

bool IsTsdOpened(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) > 0;
}

// Register for device type.
struct DeviceTypeSetRegister {
  DeviceTypeSetRegister() {
    MsContext::device_type_seter([](std::shared_ptr<MsContext> &device_type_seter) {
#if defined(ENABLE_D)
      auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
      if (enable_ge == "1") {
        device_type_seter.reset(new (std::nothrow) MsContext("ge", kAscendDevice));
      } else {
        device_type_seter.reset(new (std::nothrow) MsContext("ms", kAscendDevice));
      }
#elif defined(ENABLE_GPU)
      device_type_seter.reset(new (std::nothrow) MsContext("ms", kGPUDevice));
#else
      device_type_seter.reset(new (std::nothrow) MsContext("vm", kCPUDevice));
#endif
    });
  }
  DeviceTypeSetRegister(const DeviceTypeSetRegister &) = delete;
  DeviceTypeSetRegister &operator=(const DeviceTypeSetRegister &) = delete;
  ~DeviceTypeSetRegister() = default;
} device_type_set_regsiter;
}  // namespace context
}  // namespace mindspore
