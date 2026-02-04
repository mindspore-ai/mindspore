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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_FLAGS_H
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_FLAGS_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "runtime/hardware_abstract/visible.h"

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

class RUNTIME_HARDWARE_EXPORT GraphKernelFlags {
 public:
  static const GraphKernelFlags &GetInstance();
  static void SaveJitConfig(const std::map<std::string, std::string> &jit_config);

  // Dump all flags to json-format string
  std::string DumpAllFlags() const;

#if defined(ENABLE_AKG)
  // Check whether graph_kernel is enabled
  bool IsEnableGraphKernel() const { return opt_level > OptLevel_0; }
#else
  bool IsEnableGraphKernel() const { return false; }
#endif

  bool IsEnableKernelPacket() const;

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
   * Enable fusion for operators with dynamic shape inputs/outputs.
   */
  bool enable_dynamic_shape_fusion{false};

  /**
   * Disable Matmul post fusion in graph kernel fusion strategy, default is false.
   */
  bool disable_matmul_post_fusion{false};

  /**
   * Enable AllReduce prologue fusion in graph kernel fusion strategy, default is true.
   */
  bool enable_allreduce_prologue_fusion{true};

  /**
   * Enable AllReduce epilogue fusion in graph kernel fusion strategy, default is true.
   */
  bool enable_allreduce_epilogue_fusion{true};

  /**
   * If this flag is set to false, then some optimizations that may cause precision loss will be enabled.
   * For example, some algebraic transformations that are unsafe for low-precision floating-point computation will be
   * allowed.
   */
  bool exact_precision_mode{true};

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
   * The generator used to compile kernels, AKG or MLIR or DVM.
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
   * Usage Example: --enable_simplify_exprs_only=2.0
   */
  std::vector<std::string> enable_simplify_exprs_only;

  /**
   * Arithmetic simplify expressions to be disabled (case sensitive).
   * Usage Example: --disable_simplify_exprs=1.0,1.2
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

  /**
   * Cluster ops to run akg cce lib.
   */
  bool enable_cce_lib{false};
  std::vector<std::string> enable_cce_lib_ops;
  std::vector<std::string> enable_cce_lib_ops_only;
  std::vector<std::string> disable_cce_lib_ops;

  /**
   * The real kernel to be clustered by kernelpacket (case sensitive).
   */
  std::vector<std::string> enable_packet_ops_only;
  /**
   * The real kernel to be disabled by kernelpacket (case sensitive).
   */
  std::vector<std::string> disable_packet_ops;

  /**
   * Fusion pattern to be enabled (case sensitive).
   * Note that the "disable_fusion_pattern" will be ignored if this flag is set.
   */
  std::vector<std::string> enable_fusion_pattern_only;
  /**
   * Fusion pattern to be disabled (case sensitive).
   */
  std::vector<std::string> disable_fusion_pattern;

 private:
  GraphKernelFlags(const std::string &graph_kernel_flags, bool enable_graph_kernel)
      : flags_cache_(graph_kernel_flags), enable_graph_kernel_(enable_graph_kernel) {}

  // get the `graph_kernel_flags` and `enable_graph_kernel`
  static std::pair<std::string, bool> GetGraphKernelConfig();
  static std::map<std::string, std::string> &GetJitConfig() {
    static std::map<std::string, std::string> jit_configs{};
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

class RUNTIME_HARDWARE_EXPORT GraphKernelPassChecker {
 public:
  static GraphKernelPassChecker &GetInstance();
  GraphKernelPassChecker(const GraphKernelPassChecker &flags) = delete;
  GraphKernelPassChecker(GraphKernelPassChecker &&flags) = delete;
  GraphKernelPassChecker &operator=(const GraphKernelPassChecker &flags) = delete;
  GraphKernelPassChecker &operator=(GraphKernelPassChecker &&flags) = delete;
  ~GraphKernelPassChecker() { PassFlagsValidation(); }

  void Init();
  void SetEnablePassActive(size_t index, bool value);
  void SetDisablePassActive(size_t index, bool value);

 private:
  GraphKernelPassChecker() = default;
  void PassFlagsValidation();

  std::vector<bool> enable_pass_active_;
  std::vector<bool> disable_pass_active_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_FLAGS_H
