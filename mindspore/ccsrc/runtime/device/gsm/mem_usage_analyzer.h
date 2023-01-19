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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_MEM_USAGE_ANALYZER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_MEM_USAGE_ANALYZER_H_
#include <memory>
#include <vector>
#include <map>
#include <set>
#include "backend/common/session/kernel_graph.h"
#include "runtime/device/gsm/swap_strategy.h"
namespace mindspore {
namespace device {
class MemUsageAnalyzer {
 public:
  MemUsageAnalyzer() = default;
  ~MemUsageAnalyzer() = default;
  void Analyze(const KernelGraphPtr &graph);

  const std::vector<std::shared_ptr<MemUsageKernelInfo>> &GetMemUsageKernelInfos() const { return kernel_infos_; }

  const std::vector<std::shared_ptr<MemUsageTensorInfo>> &GetMemUsageTensorInfos() const { return tensor_infos_; }

  size_t LeastMemNeeded() const { return least_mem_; }

  const std::shared_ptr<MemUsageKernelInfo> GetMemUsageKernelInfo(size_t kid) const {
    if (kid >= kernel_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid kernel id!!!";
    }
    return kernel_infos_[kid];
  }

  const std::shared_ptr<MemUsageTensorInfo> GetMemUsageTensorInfo(size_t tid) const {
    if (tid >= tensor_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid tensor id!!!";
    }
    return tensor_infos_[tid];
  }

 private:
  void AddOutputNodeInfo(const KernelGraphPtr &graph);
  void AddKernelAndTensorInfo(const KernelGraphPtr &graph);
  size_t AddTensorInfo(const AnfNodePtr &node, size_t index, bool is_workspace = false);
  void AddFusedTensorInfo();
  bool IsGraphOutput(const AnfNodePtr &node, size_t index);
  std::map<AnfNodePtr, std::map<size_t, size_t>> kernel_input_value_tid_;
  std::map<AnfNodePtr, std::map<size_t, size_t>> kernel_input_param_tid_;
  std::map<AnfNodePtr, std::map<size_t, size_t>> kernel_output_tid_;
  std::map<AnfNodePtr, std::map<size_t, size_t>> kernel_workspace_tid_;
  std::map<AnfNodePtr, std::set<size_t>> graph_output_nodes_;
  std::vector<std::shared_ptr<MemUsageTensorInfo>> tensor_infos_;
  std::vector<std::shared_ptr<MemUsageKernelInfo>> kernel_infos_;

  size_t tensor_num_{0};
  size_t least_mem_{0};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_MEM_USAGE_ANALYZER_H_
