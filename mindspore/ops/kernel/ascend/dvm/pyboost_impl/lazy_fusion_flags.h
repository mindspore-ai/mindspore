/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_FLAGS_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_FLAGS_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "kernel/ascend/visible.h"

namespace mindspore {
constexpr unsigned int OptLevel_0 = 0;  // Disabled
constexpr unsigned int OptLevel_1 = 1;  // Basic functions
constexpr unsigned int OptLevel_2 = 2;  // Default functions

class OPS_ASCEND_API LazyFusionFlags {
 public:
  static const LazyFusionFlags &GetInstance();
  LazyFusionFlags();
  // Dump all flags to json-format string
  std::string DumpAllFlags() const;

  ~LazyFusionFlags() = default;

  /**
   * @brief Optimization level controlling the extent of optimizations applied during lazy fusion.
   *
   * - OptLevel_0: Optimization disabled.
   * - OptLevel_1: Basic optimization functions enabled.
   * - OptLevel_2: Default optimization functions enabled.
   *
   * Default value: OptLevel_0 (optimization disabled).
   */
  uint64_t opt_level{OptLevel_0};

  /**
   * @brief Dump info as human-readable text.
   *
   * Default value: false.
   */
  bool dump_as_text{false};

  /**
   * @brief All information will be dumped in this directory if dump_as_text is True.
   *
   * Default value: "./lazy_fusion_dump".
   */
  std::string dump_dir{"./lazy_fusion_dump"};

  /**
   * @brief Enables or disables synchronization mechanisms.
   *
   * When set to true, certain operations may be synchronized to ensure data consistency.
   *
   * Default value: false.
   */
  bool synchronize{false};

  /**
   * @brief Enables or disables online_tuning.
   *
   * When set to true, matmul will be optimized by online_tuning.
   *
   * Default value: false.
   */
  bool online_tuning{false};

  /**
   * @brief List of operation names that are disabled for lazy fusion.
   *
   * Operations listed here will be excluded from the lazy fusion optimization process.
   *
   * Default value: Empty vector (no operations are disabled by default).
   */
  std::vector<std::string> disable_ops;

  /**
   * @brief List of operation names that are enabled for lazy fusion.
   *
   * Operations operators listed here can be fused.
   *
   * Default value: Empty vector.
   */
  std::vector<std::string> enable_ops;

  /**
   * @brief List of operation names that are enabled for lazy fusion.
   *
   * Operations only operators listed here can be fused.
   *
   * Default value: Empty vector.
   */
  std::vector<std::string> enable_ops_only;

  /**
   * @brief Threshold value for triggering a flush operation.
   *
   * When certain metrics (e.g., operation count) reach this threshold, a flush is triggered
   * to clear caches or re-evaluate states.
   *
   * Default value: 100.
   */
  uint64_t flush_threshold{100};

 private:
  // register the flags defined above
  void RegisterFlags(std::map<std::string, std::string> *flag_map);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_FLAGS_H
