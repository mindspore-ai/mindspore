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
#include "nnacl/call_parameter.h"
#include "src/control_flow/entrance_subgraph_kernel.h"
#include "src/control_flow/exit_subgraph_kernel.h"

namespace mindspore::lite {
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

  RemoveUselessKernels(dst_kernels, &all_non_tail_subgraphs);

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
    if (!kernel::LiteKernelUtil::IsNonTailCallSubGraph(subgraph_kernel)) {
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

  // change last non-tail call property as is tail call
  reinterpret_cast<CallParameter *>((*last_non_tail_call_iter)->op_parameter())->is_tail_call = true;
  return RET_OK;
}

void ControlFlowScheduler::RemoveUselessKernels(std::vector<kernel::LiteKernel *> *dst_kernels,
                                                std::set<kernel::LiteKernel *> *useless_kernels) {
  for (auto iter = dst_kernels->begin(); iter != dst_kernels->end();) {
    if (useless_kernels->find(*iter) != useless_kernels->end()) {
      iter = dst_kernels->erase(iter);
    } else {
      iter++;
    }
  }

  for (auto &kernel : *useless_kernels) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    if (subgraph_kernel == nullptr) {
      continue;
    }
    subgraph_kernel->set_nodes({});
    delete subgraph_kernel;
  }
  useless_kernels->clear();

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
      auto partial_node = reinterpret_cast<kernel::PartialFusionKernel *>(node->kernel());
      MS_CHECK_TRUE_MSG(partial_node != nullptr, RET_ERROR, "node cast to partial node failed.");
      auto kernels = partial_node->subgraph_kernels();
      MS_CHECK_TRUE_MSG(!kernels.empty(), RET_ERROR, "partial subgraph kernels empty.");
      auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernels.back());
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

void ControlFlowScheduler::RecordSubgraphCaller(const size_t &subgraph_index, kernel::LiteKernel *partial_node) {
  if (more_than_once_called_partial_nodes_.find(subgraph_index) == more_than_once_called_partial_nodes_.end()) {
    std::set<kernel::LiteKernel *> tmp_set{partial_node};
    more_than_once_called_partial_nodes_.insert(
      std::pair<size_t, std::set<kernel::LiteKernel *>>{subgraph_index, tmp_set});
  } else {
    more_than_once_called_partial_nodes_[subgraph_index].insert(partial_node);
  }
}

kernel::SubGraphKernel *ControlFlowScheduler::CreateEntranceSubGraph(kernel::SubGraphKernel *subgraph,
                                                                     lite::Tensor *link_tensor) {
  size_t in_tensor_size = subgraph->in_tensors().size();
  std::vector<Tensor *> old_input_tensors{};
  // entrance subgraph kernel first output tensor is the first input of the corresponding exit subgraph kernel.
  std::vector<Tensor *> new_input_tensors{link_tensor};
  for (size_t i = 0; i < in_tensor_size; i++) {
    Tensor *old_tensor = subgraph->in_tensors()[i];
    old_input_tensors.push_back(old_tensor);
    auto allocator = old_tensor->allocator();
    if (allocator == nullptr && subgraph->Context() != nullptr && subgraph->desc().arch != kernel::kDelegate) {
      allocator = subgraph->Context()->allocator;
    }
    auto new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return nullptr;
    }
    src_tensors_->push_back(new_tensor);
    new_input_tensors.push_back(new_tensor);
    kernel::LiteKernelUtil::ReplaceSubGraphNodesInTensor(subgraph, old_tensor, new_tensor);
    subgraph->set_in_tensor(new_tensor, i);
  }
  auto entrance_subgraph = kernel::LiteKernelUtil::CreateSubGraphKernel(
    {}, &old_input_tensors, &new_input_tensors, kernel::kEntranceSubGraph, *context_, schema_version_);
  return entrance_subgraph;
}

kernel::SubGraphKernel *ControlFlowScheduler::CreateExitSubGraph(kernel::SubGraphKernel *subgraph,
                                                                 lite::Tensor *link_tensor) {
  size_t out_tensor_size = subgraph->out_tensors().size();
  std::vector<Tensor *> old_output_tensors{};
  // exit subgraph kernel first input tensor is the first output of the corresponding entrance subgraph kernel.
  std::vector<Tensor *> new_output_tensors{link_tensor};
  for (size_t i = 0; i < out_tensor_size; i++) {
    Tensor *old_tensor = subgraph->out_tensors()[i];
    old_output_tensors.push_back(old_tensor);
    auto allocator = old_tensor->allocator();
    if (allocator == nullptr && subgraph->Context() != nullptr && subgraph->desc().arch != kernel::kDelegate) {
      allocator = subgraph->Context()->allocator;
    }
    auto new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return nullptr;
    }
    src_tensors_->push_back(new_tensor);
    new_output_tensors.push_back(new_tensor);
    kernel::LiteKernelUtil::ReplaceSubGraphNodesOutTensor(subgraph, old_tensor, new_tensor);
    subgraph->set_out_tensor(new_tensor, i);
  }
  auto exit_subgraph = kernel::LiteKernelUtil::CreateSubGraphKernel({}, &new_output_tensors, &old_output_tensors,
                                                                    kernel::kExitSubGraph, *context_, schema_version_);
  return exit_subgraph;
}

int ControlFlowScheduler::BuildBoundaryForMultipleCalledGraph(std::vector<kernel::LiteKernel *> *dst_kernels) {
  kernel::LiteKernelUtil::FindAllInoutKernels(*dst_kernels);

  for (auto item : more_than_once_called_partial_nodes_) {
    if (item.second.size() == 1) {
      MS_LOG(DEBUG) << "subgraph call only once.";
      continue;
    }
    auto node = item.second.begin();
    kernel::PartialFusionKernel *partial = reinterpret_cast<kernel::PartialFusionKernel *>((*node)->kernel());
    MS_CHECK_TRUE_MSG(partial != nullptr, RET_ERROR, "cast to partial node failed.");
    auto aim_kernels = partial->subgraph_kernels();
    MS_CHECK_TRUE_MSG(aim_kernels.size() == 1, RET_ERROR, "partial subgraph kernels size not right.");
    auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(aim_kernels.front());
    MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "subgraph is nullptr");
    if (kernel::LiteKernelUtil::IsTailCallSubGraph(subgraph)) {
      MS_LOG(DEBUG) << "tail call graph or output graph no need to build boundary.";
      continue;
    }

    std::vector<kernel::LiteKernel *> all_call_nodes{};
    for (auto partial_node : item.second) {
      auto call_node = kernel::LiteKernelUtil::GetPartialOutputCall(partial_node);
      all_call_nodes.push_back(call_node);
    }

    // all of the caller is tail call, continue
    if (kernel::LiteKernelUtil::IsOutputSubGraph(subgraph) &&
        std::all_of(all_call_nodes.begin(), all_call_nodes.end(),
                    [](kernel::LiteKernel *call_node) { return kernel::LiteKernelUtil::IsTailCall(call_node); })) {
      MS_LOG(DEBUG) << "graph is output graph and caller is tail call, no need to build boundary.";
      continue;
    }

    // new link tensor
    auto link_tensor = new Tensor(kNumberTypeFloat32, {1});
    if (link_tensor == nullptr) {
      MS_LOG(ERROR) << "";
      return RET_NULL_PTR;
    }
    link_tensor->set_tensor_name(subgraph->name() + "_link_tensor");
    link_tensor->set_category(Category::CONST_TENSOR);
    src_tensors_->push_back(link_tensor);

    auto entrance_subgraph = CreateEntranceSubGraph(subgraph, link_tensor);
    if (entrance_subgraph == nullptr) {
      MS_LOG(ERROR) << "create entrance subgraph failed.";
      return RET_NULL_PTR;
    }
    entrance_subgraph->set_name(subgraph->name() + "_entrance");
    dst_kernels->push_back(entrance_subgraph);

    auto exit_subgraph = CreateExitSubGraph(subgraph, link_tensor);
    if (exit_subgraph == nullptr) {
      MS_LOG(ERROR) << "create exit subgraph failed.";
      return RET_NULL_PTR;
    }
    exit_subgraph->set_name(subgraph->name() + "_exit");
    dst_kernels->push_back(exit_subgraph);

    // update partial's subgraph kernels
    std::vector<kernel::LiteKernel *> subgraph_kernels{};
    subgraph_kernels.push_back(entrance_subgraph);
    subgraph_kernels.push_back(subgraph);
    subgraph_kernels.push_back(exit_subgraph);

    // record partial nodes of this subgraph.
    auto exit_subgraph_kernel = reinterpret_cast<kernel::ExitSubGraphKernel *>(exit_subgraph);
    for (auto partial_node : item.second) {
      exit_subgraph_kernel->SetPartial(partial_node);
      auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>(partial_node->kernel());
      MS_CHECK_TRUE_MSG(partial_kernel != nullptr, RET_ERROR, "cast to partial kernel failed.");
      partial_kernel->set_subgraph_kernels(subgraph_kernels);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
