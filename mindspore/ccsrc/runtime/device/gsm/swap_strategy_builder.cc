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
#include "runtime/device/gsm/swap_strategy_builder.h"
#include <memory>
#include <queue>
#include <set>
#include <functional>
#include "include/common/utils/anfalgo.h"
#include "runtime/device/gsm/swap_strategy.h"
#include "runtime/device/gsm/mem_usage_analyzer.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace device {
namespace {
template <typename T>
void CheckVectorIndex(const std::vector<T> &input, size_t index) {
  if (input.size() <= index) {
    MS_LOG_EXCEPTION << "Invalid vector index " << index << ", vector size is " << input.size();
  }
}
}  // namespace
const size_t kSwapVirtualNodeNum = 2;  // Mark graph start and end node as virtual node
void SwapStrategyBuilder::ResetState(const KernelGraphPtr &graph, const std::shared_ptr<SwapContext> &context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(context);

  context_ = context;
  total_mem_level0_ = context->hbm_mem_size_;
  total_mem_level1_ = context->ddr_mem_size_;

  kernel_num_ = graph->execution_order().size();

  mem_used_level0_.clear();
  mem_used_level0_.resize(kernel_num_, 0);
  mem_used_level1_.clear();
  mem_used_level1_.resize(kernel_num_, 0);

  span_level1_.clear();
  span_level2_.clear();
  auto tmp_queue = std::priority_queue<std::shared_ptr<Span>, std::vector<std::shared_ptr<Span>>, SpanCmp>();
  span_queue_.swap(tmp_queue);

  analyzer_ = std::make_shared<MemUsageAnalyzer>();

  kernel_actions_.clear();
  kernel_actions_.resize(kernel_num_ + kSwapVirtualNodeNum);
}

void SwapStrategyBuilder::AnalyzeGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(analyzer_);
  analyzer_->Analyze(graph);
  if (analyzer_->LeastMemNeeded() > total_mem_level0_) {
    MS_LOG(EXCEPTION) << "Need " << analyzer_->LeastMemNeeded() << " at least, but total mem is " << total_mem_level0_;
  }
}

void SwapStrategyBuilder::RecordSpan(const std::shared_ptr<MemUsageTensorInfo> &info, size_t last_index,
                                     size_t current_index, bool output_span) {
  auto dist = current_index - last_index;
  if (dist <= 1) {
    return;
  }

  MS_EXCEPTION_IF_NULL(info);
  MS_EXCEPTION_IF_NULL(context_);

  auto span = std::make_shared<Span>();
  span->tensor_id_ = info->tensor_id_;
  span->tensor_size_ = info->tensor_size_;
  span->last_index_ = last_index;
  span->current_index_ = current_index;
  span->weight_ = (dist - 1) * info->tensor_size_;
  span->output_span_ = output_span;

  bool offload_param = context_->offload_param_to_ddr_ || context_->offload_param_to_disk_;
  bool offload_checkpoint = context_->offload_checkpoint_to_ddr_ || context_->offload_checkpoint_to_disk_;
  if (offload_param && info->node_ != nullptr && info->node_->isa<Parameter>()) {
    const auto parameter = info->node_->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    if (common::AnfAlgo::IsParameterWeight(parameter)) {
      (void)offload_param_spans_.emplace_back(span);
    } else {
      span_queue_.emplace(span);
    }
  } else if (offload_checkpoint && info->node_ != nullptr && info->node_->isa<CNode>()) {
    auto cnode = info->node_->cast<CNodePtr>();
    if (cnode != nullptr && cnode->HasAttr("checkpoint")) {
      (void)offload_checkpoint_spans_.emplace_back(span);
    } else {
      span_queue_.emplace(span);
    }
  } else {
    span_queue_.emplace(span);
  }
}

void SwapStrategyBuilder::BuildSpans() {
  MS_EXCEPTION_IF_NULL(analyzer_);
  auto &tensor_infos = analyzer_->GetMemUsageTensorInfos();
  for (auto info : tensor_infos) {
    MS_EXCEPTION_IF_NULL(info);
    if (info->tensor_id_ != info->real_tensor_id_) {
      continue;
    }

    auto &used_by_kernels = info->used_by_kernels_;
    if (used_by_kernels.empty()) {
      continue;
    }

    if (info->is_inplace_tensor_) {
      for (size_t i = 0; i < used_by_kernels.size(); ++i) {
        size_t current_index = used_by_kernels[i];
        mem_used_level0_[current_index] += info->tensor_size_;
      }
      continue;
    }

    size_t last_index = used_by_kernels[0];
    CheckVectorIndex(mem_used_level0_, last_index);
    mem_used_level0_[last_index] += info->tensor_size_;

    for (size_t i = 1; i < used_by_kernels.size(); ++i) {
      size_t current_index = used_by_kernels[i];
      CheckVectorIndex(mem_used_level0_, current_index);
      mem_used_level0_[current_index] += info->tensor_size_;
      RecordSpan(info, last_index, current_index);
      last_index = current_index;
    }

    if (info->is_graph_output_) {
      RecordSpan(info, last_index, kernel_num_, true);
    }

    if (info->is_graph_input_) {
      RecordSpan(info, last_index, used_by_kernels[0] + kernel_num_);
    }
  }
}

bool SwapStrategyBuilder::EnoughSpaceForSpan(const std::shared_ptr<Span> &span, std::vector<size_t> *mem_used,
                                             size_t total_mem_size) {
  MS_EXCEPTION_IF_NULL(span);
  MS_EXCEPTION_IF_NULL(mem_used);
  CheckVectorIndex(*mem_used, kernel_num_ - 1);
  for (size_t index = span->last_index_ + 1; index < span->current_index_; ++index) {
    (*mem_used)[index % kernel_num_] += span->tensor_size_;
    if ((*mem_used)[index % kernel_num_] > total_mem_size) {
      for (size_t r_index = span->last_index_ + 1; r_index <= index; ++r_index) {
        (*mem_used)[r_index % kernel_num_] -= span->tensor_size_;
      }
      return false;
    }
  }
  return true;
}

void SwapStrategyBuilder::ClassifyOffloadSpanLevel(const std::vector<std::shared_ptr<Span>> &spans,
                                                   bool offload_to_ddr) {
  for (auto const &span : spans) {
    bool offload_to_mem_level1 = false;
    if (offload_to_ddr) {
      offload_to_mem_level1 = EnoughSpaceForSpan(span, &mem_used_level1_, total_mem_level1_);
    }

    if (offload_to_mem_level1) {
      (void)span_level1_.emplace_back(span);
    } else {
      (void)span_level2_.emplace_back(span);
    }
  }
}

void SwapStrategyBuilder::ClassifySpanLevel() {
  MS_EXCEPTION_IF_NULL(context_);
  ClassifyOffloadSpanLevel(offload_param_spans_, context_->offload_param_to_ddr_);
  offload_param_spans_.clear();
  ClassifyOffloadSpanLevel(offload_checkpoint_spans_, context_->offload_checkpoint_to_ddr_);
  offload_checkpoint_spans_.clear();

  while (!span_queue_.empty()) {
    auto span = span_queue_.top();
    bool enough = EnoughSpaceForSpan(span, &mem_used_level0_, total_mem_level0_);
    if (!enough) {
      enough = EnoughSpaceForSpan(span, &mem_used_level1_, total_mem_level1_);
      if (enough) {
        (void)span_level1_.emplace_back(span);
      } else {
        (void)span_level2_.emplace_back(span);
      }
    }
    span_queue_.pop();
  }
}

void SwapStrategyBuilder::AddTensorAction(SwapActionType action_type, size_t tensor_id, size_t kernel_id) {
  MS_EXCEPTION_IF_NULL(analyzer_);
  auto action = std::make_shared<TensorAction>();
  action->action_ = action_type;
  action->tensor_id_ = tensor_id;

  if (kernel_id > 0 && (action_type == SwapActionType::kHBM2DDR || action_type == SwapActionType::kHBM2DISK)) {
    auto kernel_info = analyzer_->GetMemUsageKernelInfo(kernel_id - 1);
    MS_EXCEPTION_IF_NULL(kernel_info);
    if (!kernel_info->update_input_) {
      action->avoid_copy_ = true;
    }
  }

  CheckVectorIndex(kernel_actions_, kernel_id);
  (void)kernel_actions_[kernel_id].emplace_back(action);
}

void SwapStrategyBuilder::AddFusedTensorSpan(const std::shared_ptr<MemUsageTensorInfo> &info, size_t start_index,
                                             size_t current_kernel_id) {
  MS_EXCEPTION_IF_NULL(analyzer_);
  MS_EXCEPTION_IF_NULL(info);
  for (auto sub_tensor_id : info->fused_tensor_ids_) {
    auto sub_tensor_info = analyzer_->GetMemUsageTensorInfo(sub_tensor_id);
    std::vector<size_t> used_before;
    std::vector<size_t> used_after;
    for (auto kid : sub_tensor_info->used_by_kernels_) {
      if (kid < start_index) {
        (void)used_before.emplace_back(kid);
      }

      if (kid >= current_kernel_id) {
        (void)used_after.emplace_back(kid);
      }
    }
    if (!used_before.empty()) {
      size_t last = used_before.size() - 1;
      CheckVectorIndex(mem_used_level0_, used_before[last]);
      mem_used_level0_[used_before[last]] += sub_tensor_info->tensor_size_;
      for (size_t i = 0; i < last; ++i) {
        CheckVectorIndex(mem_used_level0_, used_before[i]);
        mem_used_level0_[used_before[i]] += sub_tensor_info->tensor_size_;
        RecordSpan(sub_tensor_info, used_before[i], used_before[i + 1]);
      }

      auto span = std::make_shared<Span>();
      span->tensor_id_ = sub_tensor_info->tensor_id_;
      span->tensor_size_ = sub_tensor_info->tensor_size_;
      span->last_index_ = used_before[last];
      span->current_index_ = start_index;
      bool enough_space = EnoughSpaceForSpan(span, &mem_used_level1_, total_mem_level1_);
      if (enough_space) {
        AddTensorAction(SwapActionType::kHBM2DDR, span->tensor_id_, span->last_index_ + 1);
        AddTensorAction(SwapActionType::kDDR2HBM, span->tensor_id_, span->current_index_);
      } else {
        AddTensorAction(SwapActionType::kHBM2DISK, span->tensor_id_, span->last_index_ + 1);
        AddTensorAction(SwapActionType::kDISK2HBM, span->tensor_id_, span->current_index_);
      }
    }

    if (!used_after.empty()) {
      for (size_t i = 1; i < used_after.size(); ++i) {
        CheckVectorIndex(mem_used_level0_, used_after[i]);
        mem_used_level0_[used_after[i]] += sub_tensor_info->tensor_size_;
        RecordSpan(sub_tensor_info, used_after[i - 1], used_after[i]);
      }

      if (sub_tensor_info->is_graph_output_) {
        RecordSpan(sub_tensor_info, used_after[used_after.size() - 1], kernel_num_, true);
      }
    }
  }
}

void SwapStrategyBuilder::HandleFusedTensor() {
  MS_EXCEPTION_IF_NULL(analyzer_);
  auto &tensor_infos = analyzer_->GetMemUsageTensorInfos();
  for (const auto &info : tensor_infos) {
    MS_EXCEPTION_IF_NULL(info);
    if (info->node_ != nullptr) {
      continue;
    }

    auto &used_by_kernels = info->used_by_kernels_;
    if (used_by_kernels.empty()) {
      continue;
    }

    size_t current_kernel_id = used_by_kernels[0];
    std::set<size_t, std::greater<size_t>> reference_kernels_before;
    for (auto sub_tensor_id : info->fused_tensor_ids_) {
      auto sub_tensor_info = analyzer_->GetMemUsageTensorInfo(sub_tensor_id);
      for (auto kid : sub_tensor_info->used_by_kernels_) {
        if (kid >= current_kernel_id) {
          continue;
        }
        auto iter = reference_kernels_before.find(kid);
        if (iter == reference_kernels_before.end()) {
          (void)reference_kernels_before.insert(kid);
        }
      }
    }

    size_t last_index = current_kernel_id;
    size_t start_index = current_kernel_id;
    for (const auto &kid : reference_kernels_before) {
      auto span = std::make_shared<Span>();
      span->tensor_id_ = info->tensor_id_;
      span->tensor_size_ = info->tensor_size_;
      span->last_index_ = kid - 1;
      span->current_index_ = last_index;
      bool enough_space = EnoughSpaceForSpan(span, &mem_used_level0_, total_mem_level0_);
      if (!enough_space) {
        start_index = last_index;
        break;
      }
      last_index = kid;
      start_index = last_index;
    }

    AddTensorAction(SwapActionType::kAllocHBM, info->tensor_id_, start_index);

    AddFusedTensorSpan(info, start_index, current_kernel_id);
  }
}

void SwapStrategyBuilder::SpanToTensorAction() {
  for (auto span : span_level1_) {
    MS_EXCEPTION_IF_NULL(span);
    AddTensorAction(SwapActionType::kHBM2DDR, span->tensor_id_, span->last_index_ + 1);
    if (!span->output_span_) {
      AddTensorAction(SwapActionType::kDDR2HBM, span->tensor_id_, span->current_index_ % kernel_num_);
    }
  }

  for (auto span : span_level2_) {
    MS_EXCEPTION_IF_NULL(span);
    AddTensorAction(SwapActionType::kHBM2DISK, span->tensor_id_, span->last_index_ + 1);
    if (!span->output_span_) {
      AddTensorAction(SwapActionType::kDISK2HBM, span->tensor_id_, span->current_index_ % kernel_num_);
    }
  }
}

std::shared_ptr<SwapStrategy> SwapStrategyBuilder::BuildStrategy(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(analyzer_);
  auto &exec_order = graph->execution_order();
  if (exec_order.size() != kernel_num_) {
    MS_LOG(EXCEPTION) << "Kernel num error !!!";
  }

  auto strategy = std::make_shared<SwapStrategy>();
  strategy->kernel_num_ = kernel_num_;
  strategy->virtual_node_num_ = kSwapVirtualNodeNum;
  for (size_t i = 0; i < kernel_num_; ++i) {
    strategy->nodes_[i + 1] = exec_order[i];
    (void)strategy->links_.emplace_back(std::make_shared<SwapLink>(i, i + 1));
  }

  size_t logic_kernel_num = kernel_actions_.size();
  size_t action_id = logic_kernel_num;
  for (size_t i = 0; i < logic_kernel_num; ++i) {
    auto &actions = kernel_actions_[i];
    if (actions.empty()) {
      continue;
    }
    auto swap_action = std::make_shared<SwapAction>();
    swap_action->actions_ = actions;
    strategy->actions_[action_id] = swap_action;
    (void)strategy->links_.emplace_back(std::make_shared<SwapLink>(i, action_id));
    (void)strategy->links_.emplace_back(std::make_shared<SwapLink>(action_id, i + 1));
    ++action_id;
  }

  strategy->kernel_infos_ = analyzer_->GetMemUsageKernelInfos();
  strategy->tensor_infos_ = analyzer_->GetMemUsageTensorInfos();
  return strategy;
}

std::shared_ptr<SwapStrategy> SwapStrategyBuilder::Build(const KernelGraphPtr &graph,
                                                         const std::shared_ptr<SwapContext> &context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(context);
  ResetState(graph, context);

  AnalyzeGraph(graph);

  BuildSpans();

  HandleFusedTensor();

  ClassifySpanLevel();

  SpanToTensorAction();

  return BuildStrategy(graph);
}
}  // namespace device
}  // namespace mindspore
