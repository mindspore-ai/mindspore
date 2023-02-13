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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_BUILDER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_BUILDER_H_
#include <memory>
#include <vector>
#include <queue>
#include "runtime/device/gsm/swap_strategy.h"
#include "runtime/device/gsm/mem_usage_analyzer.h"
#include "include/backend/visible.h"
namespace mindspore {
namespace device {
class BACKEND_EXPORT SwapStrategyBuilder {
 public:
  SwapStrategyBuilder() = default;
  ~SwapStrategyBuilder() = default;
  std::shared_ptr<SwapStrategy> Build(const KernelGraphPtr &graph, const std::shared_ptr<SwapContext> &context);

 private:
  struct Span {
    size_t tensor_id_{0};
    size_t tensor_size_{0};
    size_t last_index_{0};
    size_t current_index_{0};
    size_t weight_{0};
    bool output_span_{false};
  };

  struct SpanCmp {
    bool operator()(const std::shared_ptr<Span> &left, const std::shared_ptr<Span> &right) {
      if (left == nullptr || right == nullptr) {
        return true;
      }
      return left->weight_ > right->weight_;
    }
  };

  std::shared_ptr<MemUsageAnalyzer> analyzer_{nullptr};
  std::shared_ptr<SwapContext> context_{nullptr};
  size_t kernel_num_{0};
  std::priority_queue<std::shared_ptr<Span>, std::vector<std::shared_ptr<Span>>, SpanCmp> span_queue_;
  std::vector<std::shared_ptr<Span>> offload_param_spans_;
  std::vector<std::shared_ptr<Span>> offload_checkpoint_spans_;
  std::vector<std::shared_ptr<Span>> span_level1_;
  std::vector<std::shared_ptr<Span>> span_level2_;
  std::vector<size_t> mem_used_level0_;
  std::vector<size_t> mem_used_level1_;
  size_t total_mem_level0_{0};
  size_t total_mem_level1_{0};
  std::vector<std::vector<std::shared_ptr<TensorAction>>> kernel_actions_;

  void ResetState(const KernelGraphPtr &graph, const std::shared_ptr<SwapContext> &context);
  void AnalyzeGraph(const KernelGraphPtr &graph);
  void BuildSpans();
  void ClassifyOffloadSpanLevel(const std::vector<std::shared_ptr<Span>> &spans, bool offload_to_ddr);
  void ClassifySpanLevel();
  void AddFusedTensorSpan(const std::shared_ptr<MemUsageTensorInfo> &info, size_t start_index,
                          size_t current_kernel_id);
  void HandleFusedTensor();
  void SpanToTensorAction();
  void RecordSpan(const std::shared_ptr<MemUsageTensorInfo> &info, size_t last_index, size_t current_index,
                  bool output_span = false);
  bool EnoughSpaceForSpan(const std::shared_ptr<Span> &span, std::vector<size_t> *mem_used, size_t total_mem_size);
  void AddTensorAction(SwapActionType action_type, size_t tensor_id, size_t kernel_id);
  std::shared_ptr<SwapStrategy> BuildStrategy(const KernelGraphPtr &graph);
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_STRATEGY_BUILDER_H_
