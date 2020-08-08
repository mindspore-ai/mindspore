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
  save_graphs_flag_ = false;
  save_graphs_path_ = ".";
  enable_dump_ = false;
  save_dump_path_ = ".";
  tsd_ref_ = 0;
  ge_ref_ = 0;
  is_multi_graph_sink_ = false;
  is_pynative_ge_init_ = false;
  enable_reduce_precision_ = true;
  auto env_device = common::GetEnv("DEVICE_ID");
  if (!env_device.empty()) {
    device_id_ = UlongToUint(std::stoul(env_device.c_str()));
  } else {
    device_id_ = 0;
  }
  backend_policy_ = policy_map_[policy];
  device_target_ = target;
  execution_mode_ = kPynativeMode;
  enable_task_sink_ = true;
  ir_fusion_flag_ = true;
  enable_hccl_ = false;
#ifdef ENABLE_DEBUGGER
  enable_mem_reuse_ = false;
#else
  enable_mem_reuse_ = true;
#endif
  enable_gpu_summary_ = true;
  precompile_only_ = false;
  auto_mixed_precision_flag_ = false;
  enable_pynative_infer_ = false;
  enable_pynative_hook_ = false;
  enable_dynamic_mem_pool_ = true;
  graph_memory_max_size_ = "0";
  variable_memory_max_size_ = "0";
  enable_loop_sink_ = target == kAscendDevice || target == kDavinciDevice;
  profiling_mode_ = false;
  profiling_options_ = "training_trace";
  check_bprop_flag_ = false;
  max_device_memory_ = kDefaultMaxDeviceMemory;
  print_file_path_ = "";
  enable_graph_kernel_ = false;
  enable_sparse_ = false;
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

std::string MsContext::backend_policy() const {
  auto res = std::find_if(
    policy_map_.begin(), policy_map_.end(),
    [&, this](const std::pair<std::string, MsBackendPolicy> &item) { return item.second == backend_policy_; });
  if (res != policy_map_.end()) {
    return res->first;
  }
  return "unknown";
}

void MsContext::set_execution_mode(int execution_mode) {
  if (execution_mode != kGraphMode && execution_mode != kPynativeMode) {
    MS_LOG(EXCEPTION) << "The execution mode is invalid!";
  }
  execution_mode_ = execution_mode;
}

bool MsContext::set_device_target(const std::string &target) {
  if (kTargetSet.find(target) == kTargetSet.end()) {
    MS_LOG(ERROR) << "invalid device target name: " << target;
    return false;
  }
  if (target == kDavinciDevice) {
    device_target_ = kAscendDevice;
  } else {
    device_target_ = target;
  }
  if (seter_) {
    seter_(device_target_);
  }
  MS_LOG(INFO) << "ms set context device target:" << target;
  return true;
}

bool MsContext::set_device_id(uint32_t device_id) {
  device_id_ = device_id;
  MS_LOG(INFO) << "ms set context device id:" << device_id;
  return true;
}

void MsContext::set_tsd_ref(const std::string &op) {
  if (op == "--") {
    tsd_ref_--;
  } else if (op == "++") {
    tsd_ref_++;
  } else {
    tsd_ref_ = 0;
  }
}

void MsContext::set_ge_ref(const std::string &op) {
  if (op == "--") {
    ge_ref_--;
  } else if (op == "++") {
    ge_ref_++;
  } else {
    ge_ref_ = 0;
  }
}
}  // namespace mindspore
