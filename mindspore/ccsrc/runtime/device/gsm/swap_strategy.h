/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_H_
#include <vector>
#include <map>
#include <memory>
#include "ir/anf.h"

namespace mindspore {
namespace device {
struct MemUsageTensorInfo {
  size_t tensor_id_{0};
  size_t real_tensor_id_{0};
  size_t tensor_size_{0};
  AnfNodePtr node_{nullptr};
  size_t index_{0};
  bool is_workspace_{false};
  bool is_graph_output_{false};
  bool is_graph_input_{false};
  bool is_inplace_tensor_{false};
  std::vector<size_t> used_by_kernels_;
  std::vector<size_t> fused_tensor_ids_;
};

struct MemUsageKernelInfo {
  bool is_comm_{false};
  bool update_input_{false};
  std::vector<size_t> input_tensors_;
  std::vector<size_t> output_tensors_;
  std::vector<size_t> workspace_tensors_;
};

enum class SwapActionType {
  kUnDefined,
  kHBM2DDR,
  kHBM2DISK,
  kDDR2HBM,
  kDISK2HBM,
  kDDR2DISK,
  kDISK2DDR,
  kAllocHBM,
};

struct TensorAction {
  SwapActionType action_;
  size_t tensor_id_{0};
  // Avoid copy if data exists in target storage and not be updated by kernel
  bool avoid_copy_{false};
};

struct SwapAction {
  std::vector<std::shared_ptr<TensorAction>> actions_;
};

struct SwapLink {
  SwapLink(size_t from, size_t to) : from_(from), to_(to) {}
  ~SwapLink() = default;
  size_t from_{0};
  size_t to_{0};
};

struct SwapStrategy {
  size_t kernel_num_{0};
  size_t virtual_node_num_{0};
  std::map<size_t, AnfNodePtr> nodes_;
  std::map<size_t, std::shared_ptr<SwapAction>> actions_;
  std::vector<std::shared_ptr<SwapLink>> links_;
  std::vector<std::shared_ptr<MemUsageTensorInfo>> tensor_infos_;
  std::vector<std::shared_ptr<MemUsageKernelInfo>> kernel_infos_;
};

class SwapContext {
 public:
  size_t hbm_mem_size_{0};
  size_t ddr_mem_size_{0};
  size_t disk_mem_size_{0};
  bool offload_param_to_ddr_{false};
  bool offload_param_to_disk_{false};
  bool offload_checkpoint_to_ddr_{false};
  bool offload_checkpoint_to_disk_{false};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_H_
