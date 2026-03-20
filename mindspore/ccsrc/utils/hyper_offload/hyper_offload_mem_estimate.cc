/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "utils/hyper_offload/hyper_offload_mem_estimate.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#include "abstract/abstract_value.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/utils/anfalgo.h"
#include "mindapi/base/type_id.h"
#include "utils/anf_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {

// Alignment constants for Ascend memory allocator compatibility (512+31 formula).
constexpr size_t kMemAlignSize = 512;
constexpr size_t kMemAlignComplement = 31;
constexpr size_t kZeroAlignSize = 1;

size_t GetAlignSize(size_t original_size) {
  if (original_size == 0) {
    return 0;
  }
  return ((original_size + kMemAlignSize + kMemAlignComplement) / kMemAlignSize) * kMemAlignSize;
}

size_t GetNodeSizeFromInfer(const AnfNodePtr &node, size_t index) {
  try {
    ShapeVector shape = common::AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
    size_t elem_size = abstract::TypeIdSize(type_id);
    if (elem_size == 0) {
      return 0;
    }
    int64_t num_elements = 1;
    for (int64_t dim : shape) {
      if (dim <= 0) {
        return 0;
      }
      num_elements *= dim;
    }
    return static_cast<size_t>(num_elements) * elem_size;
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Failed to get tensor size for node: " << node->DebugString() << ", index: " << index;
    return 0;
  }
}

size_t GetNodeSize(const AnfNodePtr &node, size_t index) {
  size_t raw_size = GetNodeSizeFromInfer(node, index);
  size_t aligned = GetAlignSize(raw_size);
  return (aligned > 0) ? aligned : GetAlignSize(kZeroAlignSize);
}

size_t GetNodeOutputCount(const AnfNodePtr &node) { return AnfUtils::GetOutputTensorNum(node); }

void UpdateLifetimeDiff(size_t start_index, size_t end_index, size_t size, std::vector<int64_t> *lifetime_diff) {
  if (start_index >= end_index || lifetime_diff->size() < end_index + 1) {
    return;
  }
  (*lifetime_diff)[start_index] += static_cast<int64_t>(size);
  (*lifetime_diff)[end_index] -= static_cast<int64_t>(size);
}

void ApplyDiffToProfile(const std::vector<int64_t> &lifetime_diff, size_t parameter_baseline,
                        std::vector<size_t> *memory_profile) {
  int64_t prefix = static_cast<int64_t>(parameter_baseline);
  for (size_t i = 0; i < memory_profile->size() && i < lifetime_diff.size(); ++i) {
    prefix += lifetime_diff[i];
    (*memory_profile)[i] = (prefix > 0) ? static_cast<size_t>(prefix) : 0;
  }
}

using NodeFirstLastMap = std::map<AnfNodePtr, std::pair<size_t, size_t>>;
using OutputLastUsageMap = std::map<std::pair<AnfNodePtr, size_t>, size_t>;

void BuildNodeFirstLastMap(const CNodePtrList &execution_order, NodeFirstLastMap *node_first_last) {
  node_first_last->clear();
  for (size_t i = 0; i < execution_order.size(); ++i) {
    const auto &node = execution_order[i];
    if (node == nullptr || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto it = node_first_last->find(node);
    if (it == node_first_last->end()) {
      (*node_first_last)[node] = {i, i};
    } else {
      it->second.second = i;
    }
  }
}

void FillLastUsageFromUserInfo(const UserInfoList &user_info_list, const std::map<AnfNodePtr, size_t> &node_to_index,
                               OutputLastUsageMap *out_last_usage) {
  for (const auto &info : user_info_list) {
    auto node = info.node.first;
    size_t out_idx = info.node.second;
    if (node_to_index.find(node) == node_to_index.end()) {
      continue;
    }
    size_t last_usage_idx = node_to_index.at(node);
    for (const auto &user : info.users) {
      auto user_node = user.first;
      auto u_it = node_to_index.find(user_node);
      if (u_it != node_to_index.end() && u_it->second > last_usage_idx) {
        last_usage_idx = u_it->second;
      }
    }
    (*out_last_usage)[{node, out_idx}] = last_usage_idx;
  }
}

void FillLastUsageFromGraphOutputs(const FuncGraphPtr &graph, const std::map<AnfNodePtr, size_t> &node_to_index,
                                   size_t graph_end_idx, OutputLastUsageMap *out_last_usage) {
  if (graph == nullptr || graph->output() == nullptr) {
    return;
  }
  try {
    const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &out : outputs) {
      auto producer = GetRealDataNodeWithIndex(out.first);
      if (producer.first != nullptr && producer.first->isa<CNode>() &&
          node_to_index.find(producer.first) != node_to_index.end()) {
        (*out_last_usage)[{producer.first, producer.second}] = graph_end_idx;
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Failed to get graph outputs for memory est: " << e.what();
  }
}

void FillLastUsageFromSummaryNodes(const KernelGraphPtr &kg, const std::map<AnfNodePtr, size_t> &node_to_index,
                                   size_t graph_end_idx, OutputLastUsageMap *out_last_usage) {
  if (kg == nullptr || !kg->summary_node_exist()) {
    return;
  }
  const auto &summary_nodes = kg->summary_nodes();
  for (const auto &node_item : summary_nodes) {
    auto origin_node = node_item.second.first;
    size_t origin_index = static_cast<size_t>(node_item.second.second);
    if (origin_node == nullptr) {
      continue;
    }
    try {
      auto resolved = common::AnfAlgo::VisitKernelWithReturnType(origin_node, origin_index, false);
      if (resolved.first != nullptr && node_to_index.find(resolved.first) != node_to_index.end()) {
        (*out_last_usage)[{resolved.first, resolved.second}] = graph_end_idx;
      }
    } catch (const std::exception &e) {
      MS_LOG(WARNING) << "Failed to resolve summary node for memory est: " << e.what();
    }
  }
}

void BuildOutputLastUsageMap(const UserInfoList &user_info_list, const FuncGraphPtr &graph,
                             const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                             OutputLastUsageMap *out_last_usage) {
  out_last_usage->clear();
  FillLastUsageFromUserInfo(user_info_list, node_to_index, out_last_usage);
  size_t graph_end_idx = (order_size > 0) ? (order_size - 1) : 0;
  FillLastUsageFromGraphOutputs(graph, node_to_index, graph_end_idx, out_last_usage);
  auto kg = graph != nullptr ? graph->cast<KernelGraphPtr>() : nullptr;
  FillLastUsageFromSummaryNodes(kg, node_to_index, graph_end_idx, out_last_usage);
}

std::pair<size_t, size_t> GetKernelFirstLastIndex(const AnfNodePtr &kernel, const NodeFirstLastMap &node_first_last,
                                                  const CNodePtrList &index_order, size_t order_size) {
  auto fl_it = node_first_last.find(kernel);
  if (fl_it != node_first_last.end()) {
    return fl_it->second;
  }
  for (size_t i = 0; i < index_order.size(); ++i) {
    if (index_order[i].get() == kernel.get()) {
      return {i, i};
    }
  }
  return {order_size, order_size};
}

const CNodePtrList &ResolveKernelsToIterate(const KernelGraphPtr &kg, const CNodePtrList &index_order) {
  if (kg != nullptr) {
    return kg->execution_order();
  }
  return index_order;
}

void AddSingleKernelOutputsToLifetimeDiff(const AnfNodePtr &kernel, size_t first_idx, size_t last_idx,
                                          const OutputLastUsageMap &out_last_usage,
                                          std::vector<int64_t> *lifetime_diff) {
  size_t output_count = GetNodeOutputCount(kernel);
  for (size_t out_idx = 0; out_idx < output_count; ++out_idx) {
    size_t size = GetNodeSize(kernel, out_idx);
    if (size == 0) {
      continue;
    }
    auto it = out_last_usage.find({kernel, out_idx});
    size_t last_usage = (it != out_last_usage.end()) ? std::max(it->second, last_idx) : last_idx;
    UpdateLifetimeDiff(first_idx, last_usage + 1, size, lifetime_diff);
  }
}

void AddOutputTensorsToLifetimeDiff(const CNodePtrList &kernels_to_iter, const NodeFirstLastMap &node_first_last,
                                    const CNodePtrList &index_order, size_t order_size,
                                    const OutputLastUsageMap &out_last_usage, std::vector<int64_t> *lifetime_diff) {
  for (const auto &kernel : kernels_to_iter) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (!AnfUtils::IsRealKernel(kernel)) {
      continue;
    }
    auto [first_idx, last_idx] = GetKernelFirstLastIndex(kernel, node_first_last, index_order, order_size);
    if (first_idx >= order_size) {
      continue;
    }
    AddSingleKernelOutputsToLifetimeDiff(kernel, first_idx, last_idx, out_last_usage, lifetime_diff);
  }
}

void CollectPeakEstimateCandidates(const UserInfoList &user_info_list,
                                   const std::map<AnfNodePtr, size_t> &node_to_index,
                                   const NodeFirstLastMap &node_first_last,
                                   std::vector<PeakEstimateCandidate> *all_candidates) {
  for (const auto &info : user_info_list) {
    if (!IsValidOffloadNode(info)) {
      continue;
    }
    auto node = info.node.first;
    auto n_it = node_to_index.find(node);
    if (n_it == node_to_index.end()) {
      continue;
    }
    auto fl_it = node_first_last.find(node);
    size_t creation_idx = (fl_it != node_first_last.end()) ? fl_it->second.first : n_it->second;
    size_t last_usage_idx = creation_idx;
    for (const auto &user : info.users) {
      auto u_it = node_to_index.find(user.first);
      if (u_it != node_to_index.end() && u_it->second > last_usage_idx) {
        last_usage_idx = u_it->second;
      }
    }
    if (last_usage_idx <= creation_idx) {
      continue;
    }
    size_t size = GetNodeSize(node, info.node.second);
    if (size == 0) {
      continue;
    }
    all_candidates->push_back({info, size, creation_idx, last_usage_idx});
  }
}

void BuildLifetimeDiffFromKernels(const CNodePtrList &kernels_to_iter, const NodeFirstLastMap &node_first_last,
                                  const CNodePtrList &index_order, size_t order_size,
                                  const OutputLastUsageMap &out_last_usage, std::vector<int64_t> *lifetime_diff) {
  lifetime_diff->assign(order_size + 1, 0);
  AddOutputTensorsToLifetimeDiff(kernels_to_iter, node_first_last, index_order, order_size, out_last_usage,
                                 lifetime_diff);
}

void BuildMemoryProfile(const UserInfoList &user_info_list, const std::map<AnfNodePtr, size_t> &node_to_index,
                        size_t order_size, size_t parameter_memory_baseline, const CNodePtrList &index_order,
                        const FuncGraphPtr &graph, std::vector<size_t> *memory_profile,
                        NodeFirstLastMap *node_first_last) {
  auto kg = graph != nullptr ? graph->cast<KernelGraphPtr>() : nullptr;
  OutputLastUsageMap out_last_usage;
  BuildOutputLastUsageMap(user_info_list, graph, node_to_index, order_size, &out_last_usage);
  BuildNodeFirstLastMap(index_order, node_first_last);

  std::vector<int64_t> lifetime_diff;
  const CNodePtrList &kernels_to_iter = ResolveKernelsToIterate(kg, index_order);
  BuildLifetimeDiffFromKernels(kernels_to_iter, *node_first_last, index_order, order_size, out_last_usage,
                               &lifetime_diff);

  memory_profile->resize(order_size);
  ApplyDiffToProfile(lifetime_diff, parameter_memory_baseline, memory_profile);
}

}  // namespace

void BuildMemoryProfileAndCandidates(const UserInfoList &user_info_list,
                                     const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                                     size_t parameter_memory_baseline, std::vector<size_t> *memory_profile,
                                     std::vector<PeakEstimateCandidate> *all_candidates,
                                     const CNodePtrList &index_order, const FuncGraphPtr &graph) {
  NodeFirstLastMap node_first_last;
  BuildMemoryProfile(user_info_list, node_to_index, order_size, parameter_memory_baseline, index_order, graph,
                     memory_profile, &node_first_last);

  all_candidates->clear();
  CollectPeakEstimateCandidates(user_info_list, node_to_index, node_first_last, all_candidates);
}

void ReduceMemoryProfile(size_t start_index, size_t end_index, size_t size, std::vector<size_t> *memory_profile) {
  if (start_index >= end_index) {
    return;
  }
  for (size_t i = start_index; i < end_index && i < memory_profile->size(); ++i) {
    (*memory_profile)[i] = ((*memory_profile)[i] >= size) ? ((*memory_profile)[i] - size) : 0;
  }
}

size_t AccumulateParamsFromInfer(const FuncGraphPtr &graph) {
  return std::accumulate(graph->parameters().begin(), graph->parameters().end(), size_t(0),
                         [](size_t sum, const AnfNodePtr &param) {
                           if (param != nullptr && param->isa<Parameter>()) {
                             return sum + GetAlignSize(GetNodeSizeFromInfer(param, 0));
                           }
                           return sum;
                         });
}

size_t ComputeTotalParameterMemory(const FuncGraphPtr &graph) {
  if (graph == nullptr) {
    return 0;
  }
  return AccumulateParamsFromInfer(graph);
}

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
