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
#include "src/extendrt/memory_offload/infer_strategy_builder.h"

namespace mindspore {
namespace lite {
namespace {
const size_t kSwapVirtualNodeNum = 2;
template <typename T>
void CheckVectorIndex(const std::vector<T> &input, size_t index) {
  if (input.size() <= index) {
    MS_LOG_EXCEPTION << "Invalid vector index " << index << ", vector size is " << input.size();
  }
}
}  // namespace

void MemoryOffloadInferStrategyBuilder::ResetState(const lite::CompileResultPtr &compile_result,
                                                   const std::shared_ptr<device::SwapContext> &context) {
  MS_EXCEPTION_IF_NULL(compile_result);
  MS_EXCEPTION_IF_NULL(context);

  context_ = context;
  total_mem_level0_ = context->hbm_mem_size_;
  total_mem_level1_ = context->cpu_mem_size_;

  kernel_num_ = compile_result->NodeSize();

  mem_used_level0_.clear();
  mem_used_level0_.resize(kernel_num_, 0);
  mem_used_level1_.clear();
  mem_used_level1_.resize(kernel_num_, 0);

  span_level1_.clear();
  span_level2_.clear();
  auto tmp_queue = std::priority_queue<std::shared_ptr<Span>, std::vector<std::shared_ptr<Span>>, SpanCmp>();
  span_queue_.swap(tmp_queue);

  kernel_actions_.clear();
  kernel_actions_.resize(kernel_num_ + kSwapVirtualNodeNum);
}

void MemoryOffloadInferStrategyBuilder::AnalyzeMemoryInfo(const lite::CompileResultPtr &compile_result) {
  auto &exec_order = compile_result->GetNodes();
  least_mem_ = SIZE_MAX;
  size_t kernel_mem = 0;
  for (size_t i = 0; i < exec_order.size(); ++i) {
    auto compile_node = exec_order[i];
    for (auto in_tensor : compile_node->GetInputs()) {
      tensor_usedby_kernel_ids_[in_tensor].insert(i);
      kernel_mem += in_tensor->Size();
    }
    for (auto out_tensor : compile_node->GetOutputs()) {
      tensor_to_kernel_id_[out_tensor] = i;
      kernel_mem += out_tensor->Size();
    }
    node_to_mem_size_[compile_node] = kernel_mem;
    if (kernel_mem > least_mem_) {
      least_mem_ = kernel_mem;
    }
    kernel_mem = 0;
  }

  auto &all_tensors = compile_result->GetTensors();
  for (size_t i = 0; i < all_tensors.size(); ++i) {
    tensor_to_index_[all_tensors[i]] = i;
  }
}

void MemoryOffloadInferStrategyBuilder::RecordSpan(const Tensor *tensor, size_t last_index, size_t current_index,
                                                   bool output_span) {
  auto dist = current_index - last_index;
  if (dist <= 1) {
    return;
  }

  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(context_);

  auto span = std::make_shared<Span>();
  MS_EXCEPTION_IF_NULL(span);
  span->tensor_id_ = tensor_to_index_[tensor];
  span->tensor_size_ = tensor->Size();
  span->last_index_ = last_index;
  span->current_index_ = current_index;
  span->weight_ = (dist - 1) * span->tensor_size_;
  span->output_span_ = output_span;

  bool offload_param = context_->offload_param_to_cpu_ || context_->offload_param_to_disk_;
  if (offload_param && tensor->category() == lite::PARAMETER) {
    (void)offload_param_spans_.emplace_back(span);
  } else {
    span_queue_.emplace(span);
  }
}

void MemoryOffloadInferStrategyBuilder::BuildSpans() {
  auto iter = tensor_usedby_kernel_ids_.begin();
  while (iter != tensor_usedby_kernel_ids_.end()) {
    auto &used_by_kernels = iter->second;
    if (used_by_kernels.empty()) {
      continue;
    }
    auto tensor = iter->first;
    size_t first_index = tensor_to_kernel_id_[tensor];
    size_t last_index = first_index;
    for (auto current_index = used_by_kernels.begin(); current_index != used_by_kernels.end(); current_index++) {
      if (first_index == *current_index) {
        continue;
      }

      RecordSpan(tensor, last_index, *current_index);
      last_index = *current_index;
    }

    // if tensor is const or parameter, then tensor data will try to store persistently
    if (tensor->category() == lite::PARAMETER || tensor->category() == lite::GRAPH_INPUT || tensor->data() != nullptr) {
      RecordSpan(tensor, last_index, first_index + kernel_num_);
    } else if (tensor->category() == lite::GRAPH_OUTPUT) {
      RecordSpan(tensor, last_index, kernel_num_, true);
    }
  }
}

void MemoryOffloadInferStrategyBuilder::ClassifySpanLevel() {
  while (!span_queue_.empty()) {
    auto span = span_queue_.top();
    bool enough = device::SwapStrategyBuilder::EnoughSpaceForSpan(span, &mem_used_level0_, total_mem_level0_);
    if (!enough) {
      enough = device::SwapStrategyBuilder::EnoughSpaceForSpan(span, &mem_used_level1_, total_mem_level1_);
      if (enough) {
        (void)span_level1_.emplace_back(span);
      } else {
        (void)span_level2_.emplace_back(span);
      }
    }
    span_queue_.pop();
  }
}

void MemoryOffloadInferStrategyBuilder::AddTensorAction(device::SwapActionType action_type, size_t tensor_id,
                                                        size_t kernel_id) {
  auto action = std::make_shared<device::TensorAction>();
  action->action_ = action_type;
  action->tensor_id_ = tensor_id;

  if (kernel_id > 0 &&
      (action_type == device::SwapActionType::kHBM2DDR || action_type == device::SwapActionType::kHBM2DISK)) {
    auto tensor = compile_result_->GetTensors()[tensor_id];
    if (tensor->category() != lite::PARAMETER) {
      action->avoid_copy_ = true;
    }
  }

  CheckVectorIndex(kernel_actions_, kernel_id);
  (void)kernel_actions_[kernel_id].emplace_back(action);
}

std::shared_ptr<device::SwapStrategy> MemoryOffloadInferStrategyBuilder::BuildStrategy(
  const lite::CompileResultPtr &compile_result) {
  MS_EXCEPTION_IF_NULL(compile_result);
  auto &exec_order = compile_result->GetNodes();

  auto strategy = std::make_shared<device::SwapStrategy>();
  strategy->kernel_num_ = kernel_num_;
  strategy->virtual_node_num_ = kSwapVirtualNodeNum;
  size_t last_index = 0;
  for (size_t i = 0; i < kernel_num_; ++i) {
    strategy->nodes_[i + 1] = exec_order[i]->GetCNode();
    (void)strategy->links_.emplace_back(std::make_shared<device::SwapLink>(last_index, i + 1));
    last_index = i + 1;
  }

  size_t logic_kernel_num = kernel_actions_.size();
  size_t action_id = logic_kernel_num;
  for (size_t i = 0; i < logic_kernel_num; ++i) {
    auto &actions = kernel_actions_[i];
    if (actions.empty()) {
      continue;
    }
    auto swap_action = std::make_shared<device::SwapAction>();
    swap_action->actions_ = actions;
    strategy->actions_[action_id] = swap_action;
    (void)strategy->links_.emplace_back(std::make_shared<device::SwapLink>(i, action_id));
    (void)strategy->links_.emplace_back(std::make_shared<device::SwapLink>(action_id, i + 1));
    ++action_id;
  }

  return strategy;
}

std::shared_ptr<device::SwapStrategy> MemoryOffloadInferStrategyBuilder::Build(
  const lite::CompileResultPtr &compile_result, const std::shared_ptr<device::SwapContext> &context) {
  MS_EXCEPTION_IF_NULL(compile_result);
  MS_EXCEPTION_IF_NULL(context);
  ResetState(compile_result, context);

  AnalyzeMemoryInfo(compile_result);

  BuildSpans();

  ClassifySpanLevel();

  device::SwapStrategyBuilder::SpanToTensorAction();

  return BuildStrategy(compile_result);
}
}  // namespace lite
}  // namespace mindspore
