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
#ifndef MINDSPORE_LITE_EXTENDRT_MEMORY_OFFLOAD_STRATEGY_BUILDER_H_
#define MINDSPORE_LITE_EXTENDRT_MEMORY_OFFLOAD_STRATEGY_BUILDER_H_
#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include "runtime/device/gsm/swap_strategy.h"
#include "runtime/device/gsm/swap_strategy_builder.h"
#include "src/extendrt/graph_compiler/compile_result_builder.h"

namespace mindspore {
namespace lite {
class MemoryOffloadInferStrategyBuilder : public device::SwapStrategyBuilder {
 public:
  MemoryOffloadInferStrategyBuilder() = default;
  ~MemoryOffloadInferStrategyBuilder() = default;
  std::shared_ptr<device::SwapStrategy> Build(const lite::CompileResultPtr &compile_result,
                                              const std::shared_ptr<device::SwapContext> &context);

 private:
  void ResetState(const lite::CompileResultPtr &compile_result, const std::shared_ptr<device::SwapContext> &context);
  void RecordSpan(const Tensor *tensor, size_t last_index, size_t current_index, bool output_span = false);
  std::shared_ptr<device::SwapStrategy> BuildStrategy(const lite::CompileResultPtr &compile_result);
  void BuildSpans();
  void AnalyzeMemoryInfo(const lite::CompileResultPtr &compile_result);
  void ClassifySpanLevel();
  void ClassifyOffloadSpanLevel(const std::vector<std::shared_ptr<Span>> &spans, bool offload_to_ddr);
  void AddTensorAction(device::SwapActionType action_type, size_t tensor_id, size_t kernel_id);

  std::shared_ptr<device::SwapContext> context_{nullptr};
  size_t prefetch_mem_size_{0};
  std::map<lite::CompileNodePtr, size_t> node_to_mem_size_;
  std::map<const Tensor *, size_t> tensor_to_index_;
  std::map<const Tensor *, size_t> tensor_to_kernel_id_;  // output tensor, node/kernel id
  std::map<const Tensor *, std::set<size_t>> tensor_usedby_kernel_ids_;
  size_t least_mem_ = SIZE_MAX;
  lite::CompileResultPtr compile_result_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_MEMORY_OFFLOAD_STRATEGY_BUILDER_H_
