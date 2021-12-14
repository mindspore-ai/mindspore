/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/control_flow/control_flow_scheduler.h"
#include <algorithm>
#include <set>
#include "src/lite_kernel_util.h"
#include "src/runtime/kernel/arm/base/partial_fusion.h"
#include "nnacl/partial_fusion_parameter.h"

namespace mindspore::lite {
bool ControlFlowScheduler::IsNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel) {
  if (subgraph_kernel == nullptr) {
    return false;
  }
  auto nodes = subgraph_kernel->nodes();
  return std::any_of(nodes.begin(), nodes.end(),
                     [](kernel::LiteKernel *node) { return kernel::LiteKernelUtil::IsNonTailCall(node); });
}

int ControlFlowScheduler::SplitNonTailCallSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernels) {
  std::set<kernel::LiteKernel *> all_non_tail_subgraphs = GetNonTailCallSubGraphs(dst_kernels);
  for (auto item : all_non_tail_subgraphs) {
    to_process_q_.push(item);
  }

  while (!to_process_q_.empty()) {
    auto cur = to_process_q_.front();
    to_process_q_.pop();
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(cur);
    if (subgraph_kernel == nullptr) {
      MS_LOG(ERROR) << "kernel is not a subgraph kernel";
      return RET_ERROR;
    }
    std::vector<kernel::LiteKernel *> new_subgraphs{};
    auto ret = SplitSingleNonTailCallSubGraph(subgraph_kernel, &new_subgraphs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SplitSingleNonTailCallSubGraph failed, ret: " << ret;
      return ret;
    }

    // append dst_kernels
    std::copy(new_subgraphs.begin(), new_subgraphs.end(), std::back_inserter(*dst_kernels));

    AppendToProcessQ(&new_subgraphs, &all_non_tail_subgraphs);
  }

  RemoveUselessKernels(dst_kernels, all_non_tail_subgraphs);

  return RecordNonTailCallLinkInfo();
}

std::set<kernel::LiteKernel *> ControlFlowScheduler::GetNonTailCallSubGraphs(
  std::vector<kernel::LiteKernel *> *dst_kernels) {
  std::set<kernel::LiteKernel *> non_tail_subgraph_kernels{};

  // found non-tail call subgraph
  for (auto &kernel : *dst_kernels) {
#ifndef DELEGATE_CLIP
    if (kernel->desc().arch == kernel::kDelegate) {
      continue;
    }
#endif
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    if (subgraph_kernel == nullptr) {
      continue;
    }
    if (!IsNonTailCallSubGraph(subgraph_kernel)) {
      continue;
    }
    non_tail_subgraph_kernels.insert(kernel);
  }
  return non_tail_subgraph_kernels;
}

int ControlFlowScheduler::SplitSingleNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel,
                                                         std::vector<kernel::LiteKernel *> *subgraph_kernels) {
  auto nodes = subgraph_kernel->nodes();

  // get the position of the last non-tail call op.
  auto is_non_tail_call_subgraph = [](kernel::LiteKernel *node) { return kernel::LiteKernelUtil::IsNonTailCall(node); };
  auto last_non_tail_call_iter = std::find_if(nodes.rbegin(), nodes.rend(), is_non_tail_call_subgraph);
  auto distance = nodes.rend() - last_non_tail_call_iter;
  if (distance == 0) {
    MS_LOG(ERROR) << "not is a non tail call subgraph.";
    return RET_ERROR;
  }

  // recode non-tail call kernels;
  non_tail_calls_.push_back(*last_non_tail_call_iter);

  // create front subgraph
  std::vector<kernel::LiteKernel *> front_subgraph_nodes{};
  for (auto iter = nodes.begin(); iter != nodes.begin() + distance; ++iter) {
    front_subgraph_nodes.push_back(*iter);
  }
  auto cur_subgraph_type = subgraph_kernel->subgraph_type();
  auto front_subgraph = kernel::LiteKernelUtil::CreateSubGraphKernel(front_subgraph_nodes, nullptr, nullptr,
                                                                     cur_subgraph_type, *context_, schema_version_);
  subgraph_kernels->push_back(front_subgraph);

  // create last subgraph
  std::vector<kernel::LiteKernel *> last_subgraph_nodes{};
  for (auto iter = nodes.begin() + distance; iter != nodes.end(); ++iter) {
    last_subgraph_nodes.push_back(*iter);
  }
  auto last_subgraph = kernel::LiteKernelUtil::CreateSubGraphKernel(last_subgraph_nodes, nullptr, nullptr,
                                                                    cur_subgraph_type, *context_, schema_version_);
  subgraph_kernels->push_back(last_subgraph);
  return RET_OK;
}

void ControlFlowScheduler::RemoveUselessKernels(std::vector<kernel::LiteKernel *> *dst_kernels,
                                                const std::set<kernel::LiteKernel *> &useless_kernels) {
  for (auto iter = dst_kernels->begin(); iter != dst_kernels->end();) {
    if (useless_kernels.find(*iter) != useless_kernels.end()) {
      iter = dst_kernels->erase(iter);
    } else {
      iter++;
    }
  }
  return;
}

void ControlFlowScheduler::AppendToProcessQ(std::vector<kernel::LiteKernel *> *new_subgraphs,
                                            std::set<kernel::LiteKernel *> *all_non_tail_subgraphs) {
  auto new_non_tail_call_subgraphs = GetNonTailCallSubGraphs(new_subgraphs);
  for (auto &item : new_non_tail_call_subgraphs) {
    if (all_non_tail_subgraphs->find(item) == all_non_tail_subgraphs->end()) {
      to_process_q_.push(item);
      all_non_tail_subgraphs->insert(item);
    }
  }
  return;
}

int ControlFlowScheduler::RecordNonTailCallLinkInfo() {
  for (auto non_tail_call : non_tail_calls_) {
    size_t non_tail_call_output_size = non_tail_call->out_tensors().size();
    auto partial_nodes = kernel::LiteKernelUtil::GetCallInputPartials(non_tail_call);
    for (auto node : partial_nodes) {
      auto partial_node = reinterpret_cast<kernel::PartialFusionKernel *>(node);
      MS_CHECK_TRUE_MSG(partial_node != nullptr, RET_ERROR, "node cast to partial node failed.");
      auto kernel = partial_node->subgraph_kernel();
      auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "partial node's subgraph kernel is nullptr.");
      MS_CHECK_TRUE_MSG(subgraph->out_tensors().size() == non_tail_call_output_size, RET_ERROR,
                        "partial inputs and corresponding call outputs size not same.");
      for (size_t i = 0; i < non_tail_call_output_size; ++i) {
        context_->SetLinkInfo(subgraph->out_tensors()[i], non_tail_call->out_tensors()[i]);
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
