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

#ifndef MINDSPORE_CCSRC_UTILS_CONTEXT_MS_CONTEXT_H_
#define MINDSPORE_CCSRC_UTILS_CONTEXT_MS_CONTEXT_H_
#include <thread>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore {

enum MsBackendPolicy {
  kMsBackendGeOnly = 0,
  kMsBackendVmOnly = 1,
  kMsBackendGePrior = 2,
  kMsBackendVmPrior = 3,
  kMsBackendMsPrior = 4,
  kMsBackendUnknown = 5,
};

const int kGraphMode = 0;
const int kPynativeMode = 1;
const char kCPUDevice[] = "CPU";
const char kGPUDevice[] = "GPU";
const char kAscendDevice[] = "Ascend";
const char kDavinciDevice[] = "Davinci";
const char KNpuLog[] = "_npu_log";
const std::set<std::string> kTargetSet = {kCPUDevice, kGPUDevice, kAscendDevice, kDavinciDevice};
// The default max available device memory is 1024GB.
const float kDefaultMaxDeviceMemory = 1024;

class MsContext {
 public:
  ~MsContext() = default;
  MsContext(const MsContext &) = delete;
  MsContext &operator=(const MsContext &) = delete;

  static std::shared_ptr<MsContext> GetInstance();

  std::string backend_policy() const;
  bool set_backend_policy(const std::string &policy);

  int execution_mode() const { return execution_mode_; }
  void set_execution_mode(int execution_mode);

  bool enable_pynative_infer() const { return enable_pynative_infer_; }
  void set_enable_pynative_infer(bool enable_pynative_infer) { enable_pynative_infer_ = enable_pynative_infer; }

  bool enable_task_sink() const { return enable_task_sink_; }

  void set_precompile_only(bool precompile_only) { precompile_only_ = precompile_only; }
  bool precompile_only() const { return precompile_only_; }

  std::string device_target() const { return device_target_; }
  bool set_device_target(const std::string &target);

  uint32_t device_id() const { return device_id_; }
  bool set_device_id(uint32_t device_id);

  bool save_graphs_flag() const { return save_graphs_flag_; }
  void set_save_graphs_flag(bool save_graphs_flag) { save_graphs_flag_ = save_graphs_flag; }

  std::string save_graphs_path() const { return save_graphs_path_; }
  void set_save_graphs_path(const std::string &save_paths) { save_graphs_path_ = save_paths; }

  bool OpenTsd();
  bool CloseTsd(bool force = false);
  bool IsTsdOpened();
  bool InitGe();
  bool FinalizeGe(bool force = false);
  bool IsGeInited();
  void set_enable_hccl(bool enable_hccl) { enable_hccl_ = enable_hccl; }
  bool enable_hccl() const { return enable_hccl_; }
  bool PynativeInitGe();

  bool ir_fusion_flag() const { return ir_fusion_flag_; }

  bool loop_sink_flag() const { return enable_loop_sink_; }
  void set_loop_sink_flag(bool enable_loop_sink) { enable_loop_sink_ = enable_loop_sink; }
  void set_enable_mem_reuse(bool enable_mem_reuse) { enable_mem_reuse_ = enable_mem_reuse; }
  bool enable_mem_reuse() const { return enable_mem_reuse_; }

  bool save_ms_model_flag() const { return save_ms_model_flag_; }
  void set_save_ms_model_flag(bool save_ms_model_flag) { save_ms_model_flag_ = save_ms_model_flag; }

  std::string save_ms_model_path() const { return save_ms_model_path_; }
  void set_save_ms_model_path(const std::string &save_ms_model_path) { save_ms_model_path_ = save_ms_model_path; }

  void set_enable_gpu_summary(bool enable_gpu_summary) { enable_gpu_summary_ = enable_gpu_summary; }
  bool enable_gpu_summary() const { return enable_gpu_summary_; }

  void set_auto_mixed_precision_flag(bool auto_mixed_precision_flag) {
    auto_mixed_precision_flag_ = auto_mixed_precision_flag;
  }
  bool auto_mixed_precision_flag() const { return auto_mixed_precision_flag_; }

  void set_enable_reduce_precision(bool flag) { enable_reduce_precision_ = flag; }
  bool enable_reduce_precision() const { return enable_reduce_precision_; }

  void set_enable_dump(bool flag) { enable_dump_ = flag; }
  bool enable_dump() const { return enable_dump_; }

  void set_save_dump_path(const std::string &path) { save_dump_path_ = path; }
  std::string save_dump_path() const { return save_dump_path_; }

  bool IsTsdOpened() const { return tsd_ref_ > 0; }

  bool is_multi_graph_sink() const { return is_multi_graph_sink_; }
  void set_is_multi_graph_sink(bool flag) { is_multi_graph_sink_ = flag; }

  void set_enable_dynamic_mem_pool(bool enable_dynamic_mem_pool) { enable_dynamic_mem_pool_ = enable_dynamic_mem_pool; }
  bool enable_dynamic_mem_pool() const { return enable_dynamic_mem_pool_; }

  void set_graph_memory_max_size(const std::string &graph_memory_max_size) {
    graph_memory_max_size_ = graph_memory_max_size;
  }

  void set_variable_memory_max_size(const std::string &variable_memory_max_size) {
    variable_memory_max_size_ = variable_memory_max_size;
  }

  void set_enable_profiling(bool flag) { profiling_mode_ = flag; }
  bool enable_profiling() const { return profiling_mode_; }

  void set_profiling_options(const std::string &options) { profiling_options_ = options; }
  std::string profiling_options() const { return profiling_options_; }
  bool check_bprop_flag() const { return check_bprop_flag_; }
  void set_check_bprop_flag(bool check_bprop_flag) { check_bprop_flag_ = check_bprop_flag; }

  float max_device_memory() const { return max_device_memory_; }
  void set_max_device_memory(float max_device_memory) { max_device_memory_ = max_device_memory; }

 private:
  MsContext(const std::string &backend_policy, const std::string &target);
  void GetGeOptions(std::map<std::string, std::string> *ge_options) const;
  void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const;
  void SetHcclOptions(std::map<std::string, std::string> *ge_options) const;

  static std::shared_ptr<MsContext> inst_context_;
  static std::map<std::string, MsBackendPolicy> policy_map_;
  MsBackendPolicy backend_policy_;
  std::string device_target_;
  uint32_t device_id_;
  int execution_mode_;
  bool enable_pynative_infer_;
  bool save_graphs_flag_;
  std::string save_graphs_path_;
  uint32_t tsd_ref_;
  uint32_t ge_ref_;
  bool enable_task_sink_;
  bool enable_hccl_;
  bool precompile_only_;
  bool ir_fusion_flag_;
  bool auto_mixed_precision_flag_;
  bool enable_reduce_precision_;
  bool enable_loop_sink_;
  bool enable_mem_reuse_;
  std::string save_ms_model_path_;
  bool save_ms_model_flag_;
  bool enable_gpu_summary_;
  bool enable_dump_;
  std::string save_dump_path_;
  bool is_multi_graph_sink_;
  bool is_pynative_ge_init_;
  bool enable_dynamic_mem_pool_;
  std::string graph_memory_max_size_;
  std::string variable_memory_max_size_;
  std::thread tdt_print_;
  bool profiling_mode_;
  std::string profiling_options_;
  bool check_bprop_flag_;
  float max_device_memory_;
};

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CONTEXT_MS_CONTEXT_H_
