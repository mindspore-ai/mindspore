/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_STRATEGY_H_

#include <utility>
#include <vector>
#include <memory>
#include "include/utils/anfalgo.h"
#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload_utils.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
struct ReplaceInfo {
  KernelWithIndex replace_start_node;
  std::vector<KernelWithIndex> replace_rest_nodes;
};
using ReplaceInfoList = std::vector<ReplaceInfo>;

struct OffloadInfo {
  KernelWithIndex data_node;
  ReplaceInfoList replace_info_list;
};
using OffloadInfoList = std::vector<OffloadInfo>;

class HyperOffloadStrategy {
 public:
  virtual ~HyperOffloadStrategy() = default;
  virtual OffloadInfoList Run(const KernelGraphPtr &kernel_graph) = 0;
};
using HyperOffloadStrategyPtr = std::unique_ptr<HyperOffloadStrategy>;

class DistanceBaseHyperOffloadStrategy : public HyperOffloadStrategy {
 public:
  OffloadInfoList Run(const KernelGraphPtr &kernel_graph) override;
};
using DistanceBaseHyperOffloadStrategyPtr = std::unique_ptr<DistanceBaseHyperOffloadStrategy>;

class OffloadInfoFilter {
 public:
  virtual ~OffloadInfoFilter() = default;
  virtual OffloadInfoList Filter(const OffloadInfoList& offload_info_list) = 0;
};

class OffloadInfoFilterByNumber : public OffloadInfoFilter {
 public:
  OffloadInfoList Filter(const OffloadInfoList& offload_info_list) override;
};
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_STRATEGY_H_
