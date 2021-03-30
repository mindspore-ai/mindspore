/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "utils/ms_context.h"
#include <thread>
#include <atomic>
#include <fstream>
#include "ir/tensor.h"
#include "utils/ms_utils.h"

namespace mindspore {
std::atomic<bool> thread_1_must_end(false);

std::shared_ptr<MsContext> MsContext::inst_context_ = nullptr;
std::map<std::string, MsBackendPolicy> MsContext::policy_map_ = {{"ge", kMsBackendGePrior},
                                                                 {"vm", kMsBackendVmOnly},
                                                                 {"ms", kMsBackendMsPrior},
                                                                 {"ge_only", kMsBackendGeOnly},
                                                                 {"vm_prior", kMsBackendVmPrior}};

MsContext::MsContext(const std::string &policy, const std::string &target) {
  set_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG, false);
  set_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH, ".");
  set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, "python");
  set_param<bool>(MS_CTX_ENABLE_DUMP, false);
  set_param<std::string>(MS_CTX_SAVE_DUMP_PATH, ".");
  set_param<std::string>(MS_CTX_ENV_CONFIG_PATH, "");
  set_param<std::string>(MS_CTX_TUNE_MODE, "NO_TUNE");
  set_param<uint32_t>(MS_CTX_TSD_REF, 0);
  set_param<uint32_t>(MS_CTX_GE_REF, 0);

  set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
  set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  set_param<bool>(MS_CTX_ENABLE_REDUCE_PRECISION, true);
  auto env_device = common::GetEnv("DEVICE_ID");
  if (!env_device.empty()) {
    try {
      uint32_t device_id = UlongToUint(std::stoul(env_device));
      set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
    } catch (std::invalid_argument &e) {
      MS_LOG(WARNING) << "Invalid DEVICE_ID env:" << env_device << ". Please set DEVICE_ID to 0-7";
      set_param<uint32_t>(MS_CTX_DEVICE_ID, 0);
    }
  } else {
    set_param<uint32_t>(MS_CTX_DEVICE_ID, 0);
  }

  set_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH, MAX_CALL_DEPTH_DEFAULT);
  set_param<std::string>(MS_CTX_DEVICE_TARGET, target);
  set_param<int>(MS_CTX_EXECUTION_MODE, kPynativeMode);
  set_param<bool>(MS_CTX_ENABLE_TASK_SINK, true);
  set_param<bool>(MS_CTX_IR_FUSION_FLAG, true);
  set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  set_param<bool>(MS_CTX_ENABLE_MEM_REUSE, true);
  set_param<bool>(MS_CTX_ENABLE_GPU_SUMMARY, true);
  set_param<bool>(MS_CTX_PRECOMPILE_ONLY, false);
  set_param<bool>(MS_CTX_ENABLE_AUTO_MIXED_PRECISION, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_HOOK, false);
  set_param<bool>(MS_CTX_ENABLE_DYNAMIC_MEM_POOL, true);
  set_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE, "0");
  set_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE, "0");
  set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, target == kAscendDevice || target == kDavinciDevice);
  set_param<bool>(MS_CTX_ENABLE_PROFILING, false);
  set_param<std::string>(MS_CTX_PROFILING_OPTIONS, "training_trace");
  set_param<bool>(MS_CTX_CHECK_BPROP_FLAG, false);
  set_param<float>(MS_CTX_MAX_DEVICE_MEMORY, kDefaultMaxDeviceMemory);
  set_param<std::string>(MS_CTX_PRINT_FILE_PATH, "");
  set_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL, false);
  set_param<bool>(MS_CTX_ENABLE_SPARSE, false);
  set_param<bool>(MS_CTX_ENABLE_PARALLEL_SPLIT, false);
  set_param<bool>(MS_CTX_GRAD_FOR_SCALAR, false);

  backend_policy_ = policy_map_[policy];
}

std::shared_ptr<MsContext> MsContext::GetInstance() {
  if (inst_context_ == nullptr) {
    MS_LOG(DEBUG) << "Create new mindspore context";
    if (device_type_seter_) {
      device_type_seter_(inst_context_);
    }
  }
  return inst_context_;
}

bool MsContext::set_backend_policy(const std::string &policy) {
  if (policy_map_.find(policy) == policy_map_.end()) {
    MS_LOG(ERROR) << "invalid backend policy name: " << policy;
    return false;
  }
  backend_policy_ = policy_map_[policy];
  MS_LOG(INFO) << "ms set context backend policy:" << policy;
  return true;
}

#ifdef ENABLE_TDTQUE
namespace py = pybind11;
acltdtChannelHandle *MsContext::CreateAclTdtChannelHandle() {
  uint32_t device_id = get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string kReceivePrefix = "TF_RECEIVE_";
  std::string channel_name = "_npu_log";
  acltdtChannelHandle *acl_handle = acltdtCreateChannel(device_id, (kReceivePrefix + channel_name).c_str());
  if (acl_handle != nullptr) {
    MS_LOG(INFO) << "Success to create acltdt handle.";
    acl_handle_ = acl_handle;
    TdtHandle::AddHandle(&acl_handle_);
  }
  return acl_handle;
}

void MsContext::DestroyAclTdtChannelHandle() {
  if (acl_handle_ == nullptr) {
    MS_LOG(INFO) << "The acl handle has been destroyed and the point is nullptr";
    return;
  }
  aclError stopStatus = acltdtStopChannel(acl_handle_);
  if (stopStatus != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed stop acl data channel and the stopStatus is " << stopStatus << std::endl;
    return;
  }
  MS_LOG(INFO) << "Succeed stop acl data channel for host queue ";

  aclError destroydStatus = acltdtDestroyChannel(acl_handle_);
  if (destroydStatus != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed destroy acl channel and the destroyStatus is " << destroydStatus << std::endl;
    return;
  }
  TdtHandle::DelHandle(&acl_handle_);
  MS_LOG(INFO) << "Succeed destroy acl channel";
}
#endif

std::string MsContext::backend_policy() const {
  auto res = std::find_if(
    policy_map_.begin(), policy_map_.end(),
    [&, this](const std::pair<std::string, MsBackendPolicy> &item) { return item.second == backend_policy_; });
  if (res != policy_map_.end()) {
    return res->first;
  }
  return "unknown";
}

bool MsContext::enable_dump_ir() const {
#ifdef ENABLE_DUMP_IR
  return true;
#else
  return false;
#endif
}

}  // namespace mindspore
