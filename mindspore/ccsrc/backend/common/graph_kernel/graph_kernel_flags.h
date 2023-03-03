/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_FLAGS_H
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_FLAGS_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "include/backend/visible.h"

namespace mindspore::graphkernel {
constexpr unsigned int OptLevel_0 = 0;  // Disabled
constexpr unsigned int OptLevel_1 = 1;  // Basic functions
constexpr unsigned int OptLevel_2 = 2;  // Default functions
constexpr unsigned int OptLevel_3 = 3;  // Experimental functions
constexpr unsigned int OptLevel_MAX = 4;

constexpr unsigned int OpLevel_0 = 0;
constexpr unsigned int OpLevel_1 = 1;
constexpr unsigned int OpLevel_2 = 2;
constexpr unsigned int OpLevel_MAX = 3;
constexpr unsigned int default_cpu_refer_tread_num = 8;

class BACKEND_EXPORT GraphKernelFlags {
 public:
  static const GraphKernelFlags &GetInstance();
  static void SaveJitConfig(const std::map<std::string, std::string> &jit_config);

  // Dump all flags to json-format string
  std::string DumpAllFlags() const;

#ifdef ENABLE_AKG
  // Check whether graph_kernel is enabled
  bool IsEnableGraphKernel() const { return opt_level > OptLevel_0; }
#else
  bool IsEnableGraphKernel() const { return false; }
#endif

  // Check whether GraphKernel supports current situation.
  void CheckSupport() const;

  GraphKernelFlags(const GraphKernelFlags &flags) = delete;
  GraphKernelFlags(GraphKernelFlags &&flags) = delete;
  GraphKernelFlags &operator=(const GraphKernelFlags &flags) = delete;
  GraphKernelFlags &operator=(GraphKernelFlags &&flags) = delete;
  ~GraphKernelFlags() = default;

  /**
   * Dump info as human-readable text.
   * A directory "graph_kernel_dump" will be created, and all information will be dumped in this directory.
   */
  bool dump_as_text{false};

  /**
   * Enable stitch fusion in graph kernel fusion strategy.
   *
   * Experimental feature, enabled by default when opt_level=3
   */
  bool enable_stitch_fusion{false};

  /**
   * Enable recompute fusion in graph kernel fusion strategy, enabled when op_level>=2.
   */
  bool enable_recompute_fusion{false};

  /**
   * Enable parallel fusion in graph kernel fusion strategy.
   *
   * Experimental feature, enabled by default when opt_level=3
   */
  bool enable_parallel_fusion{false};

  /**
   * Parallel AKG's operators by level.
   * 0: Parallel operators by local data relation analyzation with less memory influence.
   * 1: Parallel operators with global analyzation with more memory influence.
   */
  unsigned int parallel_ops_level{OpLevel_0};

  /**
   * Enable horizontal fusion in graph kernel fusion strategy, default is false.
   */
  bool enable_horizontal_fusion{false};

  /**
   * Enable auto tensor inplace in graph kernel, default is false.
   */
  bool enable_auto_tensor_inplace{false};

  /**
   * Enable dynamic batch size for akg kernels, default is false.
   */
  bool enable_dynamic_batch{false};

  /**
   * Enable low precision in data transferring between graph kernel and computing in graph kernel
   * in graph kernel.
   * Experimental feature, enabled by the enable_low_precision flag
   */
  bool enable_low_precision{false};

  /**
   * Debug mode for graph kernel.
   * Enable Debug mode for graph kernel
   */
  bool enable_debug_mode{false};

  /**
   * Enable conv tuning on mindspore lite.
   */
  bool enable_lite_conv_tuning{false};

  /**
   * Enable vectorization on akg.
   */
  bool enable_vectorization{true};

  /**
   * Expand and cluster AKG's operators by level.
   */
  unsigned int fusion_ops_level{OpLevel_0};

  /**
   * Enable recompute fusion for CSR operations.
   */
  bool enable_csr_fusion{false};

  /**
   * Optimization level, value from 0 to 3.
   * 0: Disable GraphKernel
   * 1: Enable GraphKernel with basic features only.
   * 2: Enable GraphKernel with all stable features.
   * 3: Enable GraphKernel with all experimental features.
   * The default value is OptLevel_2 when the context "enable_graph_kernel" is set,
   * but if it's also changed in "graph_kernel_flags", then the "graph_kernel_flags" will prevail.
   */
  unsigned int opt_level{0};  // defaults 0 or 2

  /**
   * Maximum number of dom ops to fuse with reduce. Valid value should be non-negative.
   * If set negative, default value(20 on GPU/CPU, 10 on Ascend) will be used.
   */
  int reduce_fuse_depth{-1};

  /**
   * Online tuning level, value from 0 to 3.
   * 0: Disable online tuning
   * 1-3: The higher level, the larger tuning space, and the more time it takes.
   */
  unsigned int online_tuning{0};

  /**
   * Cpu refer thread num for conv and graph split tuning, default is 8.
   */
  unsigned int cpu_refer_thread_num{default_cpu_refer_tread_num};

  /**
   * Threshold for detection of recopute's memory increment case, unit is byte.
   */
  int64_t recompute_increment_threshold{0};

  /**
   * Threshold for detection of recopute's memory peak case, unit is byte.
   */
  int64_t recompute_peak_threshold{0};

  /**
   * Threshold for composite ops number.
   */
  int64_t composite_op_limit_size{200};

  /**
   * AKG's operator repository file path.
   */
  std::string repository_path;

  /**
   * Target info.
   * These flags can be used for cross-compiling. Available when the device target is cpu.
   * target_os: the operating system to run kernels.
   * cpu_arch: the architecture, default value is related to the building environment (e.g. "arm" or "x86_64")
   * cpu_feature: the instruction set to be used. (e.g. "avx" or "avx512")
   * cpu_type: the cpu processor type. (e.g. "core-avx2" or "skylake-avx512")
   */
  std::string target_os{"linux"};
  std::string cpu_arch;
  std::string cpu_feature;
  std::string cpu_type;

  /**
   * Kernel Generator.
   * The generator used to compile kernels, AKG or BISHENG.
   */
  std::string kernel_generator{"AKG"};

  /**
   * Additional expanding operators (case sensitive).
   * The operators to be added into the default expanding operator list.
   */
  std::vector<std::string> enable_expand_ops;

  /**
   * Expanding operators to be enabled (case sensitive).
   * Unlike the "enable_expand_ops", the default list will be overwritten by this list.
   * Note that the "enable_expand_ops" and "disable_expand_ops" will be ignored if this flag is set.
   */
  std::vector<std::string> enable_expand_ops_only;

  /**
   * Expanding operators to be disabled (case sensitive).
   * The behavior is undefined when this list overlaps with "enable_expand_ops".
   */
  std::vector<std::string> disable_expand_ops;

  /**
   * Additional clustering operators (case sensitive).
   * The operators to be added into the default clustering operator list.
   */
  std::vector<std::string> enable_cluster_ops;

  /**
   * Clustering operators to be enabled (case sensitive).
   * Unlike the "enable_cluster_ops", the default list will be overwritten by this list.
   * Note that the "enable_cluster_ops" and "disable_cluster_ops" will be ignored if this flag is set.
   */
  std::vector<std::string> enable_cluster_ops_only;

  /**
   * Clustering operators to be disabled (case sensitive).
   * The behavior is undefined when this list overlaps with "enable_cluster_ops".
   */
  std::vector<std::string> disable_cluster_ops;

  /**
   * Arithmetic simplify expressions to be enabled (case sensitive).
   * The default list will be overwritten by this list.
   * Note that "disable_simplify_exprs" will be ignored if this flag is set.
   */
  std::vector<std::string> enable_simplify_exprs_only;

  /**
   * Arithmetic simplify expressions to be disabled (case sensitive).
   */
  std::vector<std::string> disable_simplify_exprs;

  /**
   * Passes to be enabled.
   * By default, the passes is controlled by "opt_level" and target device,
   * user can manually enable some passes by setting this flag.
   * The format is "stage_id.pass_id" or "stage_name.pass_name", which corresponds to the ir filename.
   */
  std::vector<std::string> enable_pass;

  /**
   * Passes to be disabled.
   * By default, the passes is controlled by "opt_level" and target device,
   * user can manually disable some passes by setting this flag.
   * The format is "stage_id.pass_id" or "stage_name.pass_name", which corresponds to the ir filename.
   */
  std::vector<std::string> disable_pass;

 private:
  GraphKernelFlags(const std::string &graph_kernel_flags, bool enable_graph_kernel)
      : flags_cache_(graph_kernel_flags), enable_graph_kernel_(enable_graph_kernel) {}

  // get the `graph_kernel_flags` and `enable_graph_kernel`
  static std::pair<std::string, bool> GetGraphKernelConfig();
  static std::map<std::string, std::string> &GetJitConfig() {
    static std::map<std::string, std::string> jit_configs;
    return jit_configs;
  }

  // parse and refresh the flags
  void Refresh();
  // register the flags defined above
  void RegisterFlags(std::map<std::string, std::string> *flag_map);

  // cache the flag string to check whether the flags is changed.
  std::string flags_cache_;
  // cache the enable_graph_kernel value to check whether the context is changed.
  bool enable_graph_kernel_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_FLAGS_H
