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
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include <algorithm>
#include <set>
#include "src/litert/kernel_exec_util.h"
#include "src/litert/kernel/cpu/base/partial_fusion.h"
#include "nnacl/call_parameter.h"
#include "src/control_flow/kernel/exit_subgraph_kernel.h"
#include "src/control_flow/kernel/identity_kernel.h"
#include "src/tensorlist.h"
#include "src/common/prim_inner.h"

namespace {
const constexpr int kMinNonTailCallCount = 2;
}
#endif

namespace mindspore::lite {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
int ControlFlowScheduler::Schedule(std::vector<kernel::KernelExec *> *dst_kernels) {
  auto ret = this->IsolateSameInputPartials(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "IsolateSameInputPartials failed.");
  ret = this->IsolateOutputForCallOutputGraph(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "IsolateOutputForCallOutputGraph failed");
  ret = this->IsolateInputOfMultipleCalledGraph(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "IsolateInputOfMultipleCalledGraph failed.");
  ret = this->BuildBoundaryForMultipleCalledGraph(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "BuildBoundaryForMultipleCalledGraph failed.");
  ret = this->RecordLinkInfo(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordLinkInfo failed.");
  ret = this->SplitNonTailCallSubGraphs(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "SplitNonTailCallSubGraphs failed");
  return ret;
}

int ControlFlowScheduler::SplitNonTailCallSubGraphs(std::vector<kernel::KernelExec *> *dst_kernels) {
  std::set<kernel::KernelExec *> all_non_tail_subgraphs = GetNonTailCallSubGraphs(dst_kernels);
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
    std::vector<kernel::KernelExec *> new_subgraphs{};
    auto ret = SplitSingleNonTailCallSubGraph(subgraph_kernel, &new_subgraphs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SplitSingleNonTailCallSubGraph failed, ret: " << ret;
      return ret;
    }
    // append dst_kernels
    std::copy(new_subgraphs.begin(), new_subgraphs.end(), std::back_inserter(*dst_kernels));
    // update partial_kernel_map
    for (auto &item : *partial_kernel_subgraph_index_map_) {
      auto &partial_node = item.first;
      auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>(partial_node->kernel());
      MS_CHECK_TRUE_MSG(partial_kernel != nullptr, RET_ERROR, "cast to partial kernel failed.");
      auto subgraphs = partial_kernel->subgraph_kernels();
      auto iter = std::find(subgraphs.begin(), subgraphs.end(), subgraph_kernel);
      if (iter == subgraphs.end()) {
        continue;
      }
      subgraphs.erase(iter);
      for (auto &new_subgraph : new_subgraphs) {
        subgraphs.insert(iter, new_subgraph);
      }
      partial_kernel->set_subgraph_kernels(subgraphs);
    }
    AppendToProcessQ(&new_subgraphs, &all_non_tail_subgraphs);
  }

  RemoveUselessKernels(dst_kernels, &all_non_tail_subgraphs);

  return RET_OK;
}

std::set<kernel::KernelExec *> ControlFlowScheduler::GetNonTailCallSubGraphs(
  std::vector<kernel::KernelExec *> *dst_kernels) {
  std::set<kernel::KernelExec *> non_tail_subgraph_kernels{};

  // found non-tail call subgraph
  for (auto &kernel : *dst_kernels) {
    if (kernel->desc().arch == kernel::kDelegate) {
      continue;
    }
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    if (subgraph_kernel == nullptr) {
      continue;
    }
    if (!kernel::KernelExecUtil::IsNonTailCallSubGraph(subgraph_kernel)) {
      continue;
    }
    non_tail_subgraph_kernels.insert(kernel);
  }
  return non_tail_subgraph_kernels;
}

int ControlFlowScheduler::AdjustNodesForTailCallSubGraph(std::vector<kernel::KernelExec *> *first_part_nodes,
                                                         std::vector<kernel::KernelExec *> *second_part_nodes) {
  auto tail_call = second_part_nodes->back();
  std::vector<kernel::KernelExec *> all_need_nodes{};
  std::copy(tail_call->in_kernels().begin(), tail_call->in_kernels().end(), std::back_inserter(all_need_nodes));
  auto partials = kernel::KernelExecUtil::GetCallInputPartials(tail_call);
  std::copy(partials.begin(), partials.end(), std::back_inserter(all_need_nodes));
  for (auto partial : partials) {
    for (auto input : partial->in_kernels()) {
      if (input->op_parameter()->type_ == static_cast<int>(PRIM_IDENTITY)) {
        all_need_nodes.push_back(input);
      }
    }
  }

  for (auto need : all_need_nodes) {
    if (IsContain(*second_part_nodes, need)) {
      continue;
    }
    auto is_need = [&need](kernel::KernelExec *node) { return node == need; };
    auto iter = std::find_if(first_part_nodes->begin(), first_part_nodes->end(), is_need);
    MS_CHECK_TRUE_MSG(iter != first_part_nodes->end(), RET_ERROR, "graph is not right");
    second_part_nodes->insert(second_part_nodes->begin(), *iter);
    first_part_nodes->erase(iter);
  }
  return RET_OK;
}

int ControlFlowScheduler::SplitSubGraphNodesIntoTwoParts(kernel::SubGraphKernel *subgraph_kernel,
                                                         std::vector<kernel::KernelExec *> *first_part_nodes,
                                                         std::vector<kernel::KernelExec *> *second_part_nodes) {
  auto nodes = subgraph_kernel->nodes();

  // get the position of the last non-tail call op.
  auto is_non_tail_call = [](kernel::KernelExec *node) { return kernel::KernelExecUtil::IsNonTailCall(node); };
  auto last_non_tail_call_iter = std::find_if(nodes.rbegin(), nodes.rend(), is_non_tail_call);
  auto distance = nodes.rend() - last_non_tail_call_iter;
  if (distance == 0) {
    MS_LOG(ERROR) << "not is a non tail call subgraph.";
    return RET_ERROR;
  }

  // change last non-tail call property as is tail call
  reinterpret_cast<CallParameter *>((*last_non_tail_call_iter)->op_parameter())->is_tail_call = true;

  for (auto iter = nodes.begin(); iter != nodes.begin() + distance; ++iter) {
    first_part_nodes->push_back(*iter);
  }

  for (auto iter = nodes.begin() + distance; iter != nodes.end(); ++iter) {
    second_part_nodes->push_back(*iter);
  }

  // if second part nodes contains call node, we need call node input partials and partials' inputs.
  if (kernel::KernelExecUtil::IsTailCall(second_part_nodes->back())) {
    auto ret = AdjustNodesForTailCallSubGraph(first_part_nodes, second_part_nodes);
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "AdjustNodesForTailCallSubGraph failed.");
  }
  return RET_OK;
}

int ControlFlowScheduler::SplitSingleNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel,
                                                         std::vector<kernel::KernelExec *> *subgraph_kernels) {
  std::vector<kernel::KernelExec *> first_part_nodes{};
  std::vector<kernel::KernelExec *> second_part_nodes{};

  auto ret = SplitSubGraphNodesIntoTwoParts(subgraph_kernel, &first_part_nodes, &second_part_nodes);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "SplitSubGraphNodesIntoTwoParts failed.");

  auto cur_subgraph_type = subgraph_kernel->subgraph_type();
  auto first_subgraph = kernel::KernelExecUtil::CreateSubGraphKernel(first_part_nodes, nullptr, nullptr,
                                                                     cur_subgraph_type, *context_, schema_version_);
  subgraph_kernels->push_back(first_subgraph);

  auto second_subgraph = kernel::KernelExecUtil::CreateSubGraphKernel(second_part_nodes, nullptr, nullptr,
                                                                      cur_subgraph_type, *context_, schema_version_);
  subgraph_kernels->push_back(second_subgraph);
  return RET_OK;
}

void ControlFlowScheduler::RemoveUselessKernels(std::vector<kernel::KernelExec *> *dst_kernels,
                                                std::set<kernel::KernelExec *> *useless_kernels) {
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

void ControlFlowScheduler::AppendToProcessQ(std::vector<kernel::KernelExec *> *new_subgraphs,
                                            std::set<kernel::KernelExec *> *all_non_tail_subgraphs) {
  auto new_non_tail_call_subgraphs = GetNonTailCallSubGraphs(new_subgraphs);
  for (auto &item : new_non_tail_call_subgraphs) {
    if (all_non_tail_subgraphs->find(item) == all_non_tail_subgraphs->end()) {
      to_process_q_.push(item);
      all_non_tail_subgraphs->insert(item);
    }
  }
  return;
}

int ControlFlowScheduler::RecordNonTailCallLinkInfo(kernel::KernelExec *non_tail_call) {
  size_t non_tail_call_output_size = non_tail_call->out_tensors().size();
  auto partial_nodes = kernel::KernelExecUtil::GetCallInputPartials(non_tail_call);
  for (auto node : partial_nodes) {
    auto partial_node = reinterpret_cast<kernel::PartialFusionKernel *>(node->kernel());
    MS_CHECK_TRUE_MSG(partial_node != nullptr, RET_ERROR, "node cast to partial node failed.");
    auto kernels = partial_node->subgraph_kernels();
    MS_CHECK_TRUE_MSG(!kernels.empty(), RET_ERROR, "partial subgraph kernels empty.");
    auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernels.back());
    MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "partial node's subgraph kernel is nullptr.");
    if (kernel::KernelExecUtil::IsTailCallSubGraph(subgraph)) {
      std::queue<kernel::KernelExec *> tail_call_q{};
      tail_call_q.push(subgraph->out_nodes().front());
      std::vector<kernel::KernelExec *> final_graphs{};
      std::set<kernel::KernelExec *> reviewed_graphs{};
      auto ret = GetTailCallFinalSubgraphs(&tail_call_q, &final_graphs, reviewed_graphs);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "GetTailCallFinalSubgraphs failed.");
      for (auto item : final_graphs) {
        MS_CHECK_TRUE_MSG(item->out_tensors().size() == non_tail_call_output_size, RET_ERROR,
                          "subgraph outputs and corresponding call outputs size not same.");
        for (size_t i = 0; i < non_tail_call_output_size; ++i) {
          context_->SetLinkInfo(item->out_tensors()[i], non_tail_call->out_tensors()[i]);
        }
      }
    } else {
      MS_CHECK_TRUE_MSG(subgraph->out_tensors().size() == non_tail_call_output_size, RET_ERROR,
                        "partial inputs and corresponding call outputs size not same.");
      for (size_t i = 0; i < non_tail_call_output_size; ++i) {
        context_->SetLinkInfo(subgraph->out_tensors()[i], non_tail_call->out_tensors()[i]);
      }
    }
  }
  return RET_OK;
}

int ControlFlowScheduler::RecordAllNonTailCallLinkInfo(std::vector<kernel::KernelExec *> *dst_kernels) {
  for (auto dst_kernel : *dst_kernels) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(dst_kernel);
    MS_CHECK_TRUE_MSG(subgraph_kernel != nullptr, RET_ERROR, "node cast to subgraph kernel failed.");
    for (auto node : subgraph_kernel->nodes()) {
      if (kernel::KernelExecUtil::IsNonTailCall(node)) {
        non_tail_calls_.push_back(node);
      }
    }
  }

  for (auto non_tail_call : non_tail_calls_) {
    auto ret = RecordNonTailCallLinkInfo(non_tail_call);
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordNonTailCallLinkInfo, failed");
  }
  return RET_OK;
}

void ControlFlowScheduler::RecordSubgraphCaller(const size_t &subgraph_index, kernel::KernelExec *partial_node) {
  if (more_than_once_called_partial_nodes_.find(subgraph_index) == more_than_once_called_partial_nodes_.end()) {
    std::set<kernel::KernelExec *> tmp_set{partial_node};
    more_than_once_called_partial_nodes_.insert(
      std::pair<size_t, std::set<kernel::KernelExec *>>{subgraph_index, tmp_set});
  } else {
    more_than_once_called_partial_nodes_[subgraph_index].insert(partial_node);
  }
}

kernel::SubGraphKernel *ControlFlowScheduler::CreateEntranceSubGraph(kernel::SubGraphKernel *subgraph,
                                                                     lite::Tensor *link_tensor) {
  if (subgraph == nullptr || link_tensor == nullptr) {
    MS_LOG(ERROR) << "input is nullptr.";
    return nullptr;
  }
  size_t in_tensor_size = subgraph->in_tensors().size();
  std::vector<Tensor *> old_input_tensors{};
  // entrance subgraph kernel first output tensor is the first input of the corresponding exit subgraph kernel.
  std::vector<Tensor *> new_input_tensors{link_tensor};
  for (size_t i = 0; i < in_tensor_size; i++) {
    Tensor *old_tensor = subgraph->in_tensors()[i];
    old_input_tensors.push_back(old_tensor);
    auto allocator = old_tensor->allocator();
    auto new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return nullptr;
    }
    src_tensors_->push_back(new_tensor);
    new_input_tensors.push_back(new_tensor);
    kernel::KernelExecUtil::ReplaceSubGraphNodesInTensor(subgraph, old_tensor, new_tensor);
    subgraph->set_in_tensor(new_tensor, i);
  }
  auto entrance_subgraph = kernel::KernelExecUtil::CreateSubGraphKernel(
    {}, &old_input_tensors, &new_input_tensors, kernel::kEntranceSubGraph, *context_, schema_version_);
  return entrance_subgraph;
}

kernel::SubGraphKernel *ControlFlowScheduler::CreateExitSubGraph(kernel::SubGraphKernel *subgraph,
                                                                 lite::Tensor *link_tensor) {
  if (subgraph == nullptr || link_tensor == nullptr) {
    MS_LOG(ERROR) << "input is nullptr.";
    return nullptr;
  }
  size_t out_tensor_size = subgraph->out_tensors().size();
  std::vector<Tensor *> old_output_tensors{};
  // exit subgraph kernel first input tensor is the first output of the corresponding entrance subgraph kernel.
  std::vector<Tensor *> new_output_tensors{link_tensor};
  for (size_t i = 0; i < out_tensor_size; i++) {
    Tensor *old_tensor = subgraph->out_tensors()[i];
    old_output_tensors.push_back(old_tensor);
    auto allocator = old_tensor->allocator();
    auto new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return nullptr;
    }
    src_tensors_->push_back(new_tensor);
    new_output_tensors.push_back(new_tensor);
    kernel::KernelExecUtil::ReplaceSubGraphNodesOutTensor(subgraph, old_tensor, new_tensor);
    subgraph->set_out_tensor(new_tensor, i);
  }
  auto exit_subgraph = kernel::KernelExecUtil::CreateSubGraphKernel({}, &new_output_tensors, &old_output_tensors,
                                                                    kernel::kExitSubGraph, *context_, schema_version_);
  return exit_subgraph;
}

kernel::SubGraphKernel *ControlFlowScheduler::AddOutputKernel(kernel::SubGraphKernel *subgraph) {
  auto inputs = subgraph->in_tensors();
  auto outputs = subgraph->out_tensors();
  auto nodes = subgraph->nodes();

  auto call_node = subgraph->out_nodes().front();
  reinterpret_cast<CallParameter *>(call_node->op_parameter())->is_tail_call = false;

  size_t out_tensor_size = call_node->out_tensors().size();
  std::vector<Tensor *> old_output_tensors{};
  std::vector<Tensor *> new_output_tensors{};
  for (size_t i = 0; i < out_tensor_size; i++) {
    Tensor *old_tensor = subgraph->out_tensors()[i];
    old_output_tensors.push_back(old_tensor);
    auto allocator = old_tensor->allocator();
    auto new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed.";
      return nullptr;
    }
    src_tensors_->push_back(new_tensor);
    new_output_tensors.push_back(new_tensor);
    kernel::KernelExecUtil::ReplaceSubGraphNodesOutTensor(subgraph, old_tensor, new_tensor);
    call_node->set_out_tensor(new_tensor, i);
    context_->ReplaceLinkInfoReceiverWithNewOne(new_tensor, old_tensor);
  }
  auto output_node = kernel::IdentityKernel::Create(new_output_tensors, old_output_tensors, this->context_);
  output_node->set_name(call_node->name() + "_output");
  kernel::KernelKey output_desc = call_node->desc();
  output_desc.type = PrimType_Inner_Identity;
  output_node->set_desc(output_desc);
  output_node->AddInKernel(call_node);
  call_node->AddOutKernel(output_node);
  nodes.push_back(output_node);
  auto subgraph_type = subgraph->subgraph_type();
  auto new_subgraph =
    kernel::KernelExecUtil::CreateSubGraphKernel(nodes, &inputs, &outputs, subgraph_type, *context_, schema_version_);
  return new_subgraph;
}

int ControlFlowScheduler::GetSubGraphsWhichNeedBoundary() {
  // among the more than once call subgraphs, if one of it's corresponding partial nodes' call node is non-tail call.
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

    std::vector<kernel::KernelExec *> all_call_nodes{};
    for (auto partial_node : item.second) {
      auto call_node = kernel::KernelExecUtil::GetPartialOutputCall(partial_node);
      if (call_node == nullptr) {
        MS_LOG(ERROR) << "call_node is nullptr.";
        return RET_ERROR;
      }
      all_call_nodes.push_back(call_node);
    }

    // non-tail call size less than 2, continue
    int non_tail_call_size = 0;
    for (auto call_node : all_call_nodes) {
      if (kernel::KernelExecUtil::IsNonTailCall(call_node)) {
        non_tail_call_size++;
      }
    }
    if (non_tail_call_size < kMinNonTailCallCount) {
      MS_LOG(DEBUG) << "no need to build boundary.";
      continue;
    }
    for (auto partial_node : item.second) {
      subgraphs_need_boundary_[subgraph].insert(partial_node);
    }
  }
  return RET_OK;
}

int ControlFlowScheduler::BuildBoundaryForMultipleCalledGraph(std::vector<kernel::KernelExec *> *dst_kernels) {
  for (auto &item : subgraphs_need_boundary_) {
    auto subgraph = item.first;
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
    std::vector<kernel::KernelExec *> subgraph_kernels{};
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

int ControlFlowScheduler::IsolateOutputForCallOutputGraph(std::vector<kernel::KernelExec *> *dst_kernels) {
  kernel::KernelExec *main_graph_kernel = dst_kernels->front();
  if (!kernel::KernelExecUtil::IsOutputSubGraph(main_graph_kernel)) {
    MS_LOG(DEBUG) << "Not is output graph.";
    return RET_OK;
  }

  auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(main_graph_kernel);
  MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "cast to subgraph failed.");
  if (!(subgraph->out_nodes().size() == 1 && subgraph->out_nodes().front()->type() == schema::PrimitiveType_Call)) {
    MS_LOG(DEBUG) << "main graph output is not call node.";
    return RET_OK;
  }

  auto new_subgraph = AddOutputKernel(subgraph);
  MS_CHECK_TRUE_MSG(new_subgraph != nullptr, RET_ERROR, "create output subgraph failed.");
  new_subgraph->set_name(subgraph->name());
  std::replace(dst_kernels->begin(), dst_kernels->end(), subgraph, new_subgraph);

  subgraph->set_nodes({});
  delete subgraph;
  return RET_OK;
}

int ControlFlowScheduler::GetTailCallFinalSubgraphs(std::queue<kernel::KernelExec *> *tail_call_q,
                                                    std::vector<kernel::KernelExec *> *final_graphs,
                                                    std::set<kernel::KernelExec *> reviewed_graphs) {
  if (tail_call_q->empty()) {
    return RET_OK;
  }
  auto tail_call = tail_call_q->front();
  tail_call_q->pop();
  auto partials = kernel::KernelExecUtil::GetCallInputPartials(tail_call);
  for (auto partial : partials) {
    auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>(partial->kernel());
    MS_CHECK_TRUE_MSG(partial_kernel != nullptr, RET_ERROR, "cast to partial kernel failed.");
    // only get the output subgraph, the last subgraph is the output subgraph.
    auto subgraphs = partial_kernel->subgraph_kernels();
    auto subgraph = subgraphs.back();
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(subgraph);
    MS_CHECK_TRUE_MSG(subgraph_kernel != nullptr, RET_ERROR, "cast to subgraph kernel failed.");
    if (kernel::KernelExecUtil::IsTailCallSubGraph(subgraph_kernel)) {
      if (reviewed_graphs.find(subgraph_kernel) == reviewed_graphs.end()) {
        tail_call_q->push(subgraph_kernel->out_nodes().front());
      }
    } else {
      final_graphs->push_back(subgraph);
    }
    reviewed_graphs.insert(subgraph);
  }
  return GetTailCallFinalSubgraphs(tail_call_q, final_graphs, reviewed_graphs);
}

int ControlFlowScheduler::RecordTailCallLinkInfo(kernel::KernelExec *tail_call) {
  std::queue<kernel::KernelExec *> tail_call_q{};
  tail_call_q.push(tail_call);
  std::vector<kernel::KernelExec *> final_graphs{};
  std::set<kernel::KernelExec *> reviewed_graphs{};
  auto ret = GetTailCallFinalSubgraphs(&tail_call_q, &final_graphs, reviewed_graphs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetTailCallFinalSubgraphs failed.";
    return ret;
  }

  if (std::any_of(final_graphs.begin(), final_graphs.end(), [&tail_call](kernel::KernelExec *item) {
        return item->out_tensors().size() != tail_call->out_tensors().size();
      })) {
    MS_LOG(DEBUG) << "not is mindir model, return ok.";
    return RET_OK;
  }

  for (auto final_graph : final_graphs) {
    for (size_t i = 0; i < final_graph->out_tensors().size(); ++i) {
      context_->SetLinkInfo(final_graph->out_tensors()[i], tail_call->out_tensors()[i]);
    }
  }
  return RET_OK;
}

int ControlFlowScheduler::RecordAllTailCallLinkInfo(std::vector<kernel::KernelExec *> *dst_kernels) {
  std::vector<kernel::KernelExec *> all_tail_calls{};
  for (auto dst_kernel : *dst_kernels) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(dst_kernel);
    if (kernel::KernelExecUtil::IsTailCallSubGraph(subgraph_kernel)) {
      all_tail_calls.push_back(subgraph_kernel->out_nodes().front());
    }
  }

  for (auto tail_call : all_tail_calls) {
    auto ret = RecordTailCallLinkInfo(tail_call);
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordTailCallLinkInfo, failed");
  }
  return RET_OK;
}

kernel::SubGraphKernel *ControlFlowScheduler::IsolatePartialInputs(kernel::SubGraphKernel *subgraph,
                                                                   kernel::KernelExec *partial) {
  auto inputs = subgraph->in_tensors();
  auto outputs = subgraph->out_tensors();
  auto nodes = subgraph->nodes();

  auto old_partial_inputs = partial->in_tensors();

  std::vector<Tensor *> new_partial_inputs{};
  for (size_t i = 0; i < old_partial_inputs.size(); i++) {
    Tensor *old_tensor = old_partial_inputs[i];
    auto allocator = old_tensor->allocator();
    Tensor *new_tensor = nullptr;
    if (old_tensor->data_type() == kObjectTypeTensorType) {
      auto old_tensor_list = reinterpret_cast<TensorList *>(old_tensor);
      new_tensor = TensorList::CopyTensorList(*old_tensor_list, false, allocator);
    } else {
      new_tensor = Tensor::CopyTensor(*old_tensor, false, allocator);
    }
    MS_CHECK_TRUE_MSG(new_tensor != nullptr, nullptr, "new tensor failed.");
    new_tensor->set_category(VAR);
    partial->set_in_tensor(new_tensor, i);
    src_tensors_->push_back(new_tensor);
    new_partial_inputs.push_back(new_tensor);
  }
  auto identity_node = kernel::IdentityKernel::Create(old_partial_inputs, new_partial_inputs, this->context_);
  identity_node->set_name(partial->name() + "_input_identity");
  kernel::KernelKey identity_desc = partial->desc();
  identity_desc.type = PrimType_Inner_Identity;
  identity_node->set_desc(identity_desc);
  // update identity and partial in kernels and out kernels
  for (auto partial_in_kernel : partial->in_kernels()) {
    auto output_kernels = partial_in_kernel->out_kernels();
    std::replace(output_kernels.begin(), output_kernels.end(), partial, identity_node);
    partial_in_kernel->set_out_kernels(output_kernels);
    identity_node->AddInKernel(partial_in_kernel);
  }
  identity_node->AddOutKernel(partial);
  partial->set_in_kernels({identity_node});
  auto partial_iter = std::find(nodes.begin(), nodes.end(), partial);
  nodes.insert(partial_iter, identity_node);
  auto subgraph_type = subgraph->subgraph_type();
  auto new_subgraph =
    kernel::KernelExecUtil::CreateSubGraphKernel(nodes, &inputs, &outputs, subgraph_type, *context_, schema_version_);
  return new_subgraph;
}

std::set<kernel::KernelExec *> ControlFlowScheduler::GetSameInputPartials() {
  std::unordered_map<Tensor *, std::set<kernel::KernelExec *>> input_partial_pairs{};
  for (auto item : *partial_kernel_subgraph_index_map_) {
    for (auto input : item.first->in_tensors()) {
      if (input_partial_pairs.find(input) == input_partial_pairs.end()) {
        std::set<kernel::KernelExec *> partials{};
        partials.insert(item.first);
        input_partial_pairs[input] = partials;
      } else {
        input_partial_pairs[input].insert(item.first);
      }
    }
  }

  std::set<kernel::KernelExec *> same_input_partials{};
  for (auto item : input_partial_pairs) {
    if (item.second.size() > 1) {
      for (auto partial : item.second) {
        same_input_partials.insert(partial);
      }
    }
  }
  return same_input_partials;
}

int ControlFlowScheduler::IsolateSameInputPartials(std::vector<kernel::KernelExec *> *dst_kernels) {
  auto same_input_partials = GetSameInputPartials();

  for (auto partial : same_input_partials) {
    auto subgraph = kernel::KernelExecUtil::BelongToWhichSubGraph(*dst_kernels, partial);
    MS_CHECK_TRUE_MSG(subgraph != nullptr, RET_ERROR, "can not find belong graph.");
    kernel::SubGraphKernel *new_subgraph = IsolatePartialInputs(subgraph, partial);
    MS_CHECK_TRUE_MSG(new_subgraph != nullptr, RET_ERROR, "create new subgraph failed.");
    new_subgraph->set_name(subgraph->name());

    std::replace(dst_kernels->begin(), dst_kernels->end(), subgraph, new_subgraph);
    UpdateSubGraphMap(new_subgraph, subgraph);

    subgraph->set_nodes({});
    delete subgraph;
  }

  SetSubgraphForPartialNode(partial_kernel_subgraph_index_map_, subgraph_index_subgraph_kernel_map_);
  return RET_OK;
}

int ControlFlowScheduler::IsolateInputOfMultipleCalledGraph(std::vector<kernel::KernelExec *> *dst_kernels) {
  auto ret = GetSubGraphsWhichNeedBoundary();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "GetSubGraphsWhichNeedBoundary failed.");
  std::unordered_map<kernel::SubGraphKernel *, kernel::SubGraphKernel *> replace_pair{};

  for (auto &item : subgraphs_need_boundary_) {
    auto subgraph = item.first;
    std::vector<kernel::KernelExec *> input_partials{};
    for (auto input : subgraph->in_nodes()) {
      MS_CHECK_TRUE_MSG(input->op_parameter() != nullptr, RET_ERROR, "op_parameter is nullptr.");
      if (input->op_parameter()->type_ == static_cast<int>(schema::PrimitiveType_PartialFusion)) {
        input_partials.push_back(input);
      }
    }
    kernel::SubGraphKernel *new_subgraph = nullptr;
    kernel::SubGraphKernel *cur_subgraph = subgraph;
    for (auto cur_partial : input_partials) {
      new_subgraph = IsolatePartialInputs(cur_subgraph, cur_partial);
      MS_CHECK_TRUE_MSG(new_subgraph != nullptr, RET_ERROR, "create new subgraph failed.");
      new_subgraph->set_name(cur_subgraph->name());

      cur_subgraph->set_nodes({});
      delete cur_subgraph;
      cur_subgraph = new_subgraph;
    }

    if (new_subgraph != nullptr) {
      replace_pair[subgraph] = new_subgraph;
    }
  }

  // update all partial nodes' subgraph
  for (auto item : replace_pair) {
    auto old_subgrpah = item.first;
    auto new_subgraph = item.second;
    for (auto partial_node : subgraphs_need_boundary_[old_subgrpah]) {
      auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>(partial_node->kernel());
      MS_CHECK_TRUE_MSG(partial_kernel != nullptr, RET_ERROR, "cast to partial kernel failed.");
      partial_kernel->set_subgraph_kernels({new_subgraph});
      subgraphs_need_boundary_[new_subgraph].insert(partial_node);
    }
  }

  for (auto item : replace_pair) {
    auto old_subgrpah = item.first;
    subgraphs_need_boundary_.erase(old_subgrpah);
  }

  // update all dst_kernels
  for (auto item : replace_pair) {
    auto old_subgrpah = item.first;
    auto new_subgraph = item.second;
    std::replace(dst_kernels->begin(), dst_kernels->end(), old_subgrpah, new_subgraph);
  }

  return RET_OK;
}

void ControlFlowScheduler::SetSubgraphForPartialNode(
  std::unordered_map<kernel::KernelExec *, size_t> *partial_kernel_subgraph_index_map,
  std::unordered_map<size_t, kernel::KernelExec *> *subgraph_index_subgraph_kernel_map) {
  partial_kernel_subgraph_index_map_ = partial_kernel_subgraph_index_map;
  subgraph_index_subgraph_kernel_map_ = subgraph_index_subgraph_kernel_map;

  for (auto &pair : *partial_kernel_subgraph_index_map) {
    auto partial_kernel = static_cast<kernel::PartialFusionKernel *>((pair.first)->kernel());
    auto &subgraph_index = pair.second;
    partial_kernel->set_subgraph_kernels({subgraph_index_subgraph_kernel_map->at(subgraph_index)});
  }
}

void ControlFlowScheduler::UpdateSubGraphMap(kernel::KernelExec *new_subgraph, kernel::KernelExec *old_subgraph) {
  for (auto &item : *subgraph_index_subgraph_kernel_map_) {
    if (item.second == old_subgraph) {
      item.second = new_subgraph;
    }
  }
  return;
}

int ControlFlowScheduler::RecordLinkInfo(std::vector<kernel::KernelExec *> *dst_kernels) {
  auto ret = RecordPartialInputLinkInfo();
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordPartialInputLinkInfo failed.");
  ret = this->RecordAllTailCallLinkInfo(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordAllTailCallLinkInfo failed");
  ret = this->RecordAllNonTailCallLinkInfo(dst_kernels);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "RecordAllNonTailCallLinkInfo failed");
  return RET_OK;
}

int ControlFlowScheduler::RecordPartialInputLinkInfo() {
  for (auto &pair : *partial_kernel_subgraph_index_map_) {
    auto partial_kernel = reinterpret_cast<kernel::PartialFusionKernel *>((pair.first)->kernel());
    MS_CHECK_TRUE_MSG(partial_kernel != nullptr, RET_ERROR, "cast to partial kernel failed.");
    auto subgraph_kernels = partial_kernel->subgraph_kernels();
    MS_CHECK_TRUE_MSG(!subgraph_kernels.empty(), RET_ERROR, "partial corresponding subgraph kernels empty.");
    auto subgraph_kernel = subgraph_kernels.front();
    MS_CHECK_TRUE_MSG(partial_kernel->in_tensors().size() == subgraph_kernel->in_tensors().size(), RET_ERROR,
                      "partial inputs and corresponding subgraph inputs size not same.");
    for (size_t i = 0; i < partial_kernel->in_tensors().size(); ++i) {
      context_->SetLinkInfo(partial_kernel->in_tensors()[i], subgraph_kernel->in_tensors()[i]);
    }
  }
  return RET_OK;
}

#else
int ControlFlowScheduler::Schedule(std::vector<kernel::KernelExec *> *dst_kernels) { return RET_OK; }
void ControlFlowScheduler::SetSubgraphForPartialNode(
  std::unordered_map<kernel::KernelExec *, size_t> *partial_kernel_subgraph_index_map,
  std::unordered_map<size_t, kernel::KernelExec *> *subgraph_index_subgraph_kernel_map) {
  return;
}
void ControlFlowScheduler::RecordSubgraphCaller(const size_t &subgraph_index, kernel::KernelExec *partial_node) {
  return;
}
#endif
}  // namespace mindspore::lite
