/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/add_atomic_clean.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
#include "kernel/kernel.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/backend/kernel_graph.h"
#include "include/common/debug/anf_ir_dump.h"
#include "kernel/common_utils.h"

namespace mindspore::graphkernel {
namespace {
std::set<int64_t> GetUniqReduceAxes(const AnfNodePtr &node, bool is_ascend = false) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceSum)) {
    MS_LOG(EXCEPTION) << "Expect ReduceSum node, but got " << common::AnfAlgo::GetCNodeName(node);
  }

  auto input = node->cast<CNodePtr>()->input(kFirstDataInputIndex);
  ShapeVector src_shape_vec;
  if (is_ascend) {
    src_shape_vec = GetDeviceShape(input);
  } else {
    src_shape_vec = GetShape(input);
  }
  auto axis_vec = GetReduceAxis(node);
  if (axis_vec.empty()) {
    axis_vec.resize(src_shape_vec.size());
    for (size_t i = 0; i < src_shape_vec.size(); ++i) {
      axis_vec[i] = SizeToLong(i);
    }
  } else {
    (void)std::transform(axis_vec.begin(), axis_vec.end(), axis_vec.begin(), [&src_shape_vec](int64_t axis) -> int64_t {
      return axis < 0 ? axis + SizeToLong(src_shape_vec.size()) : axis;
    });
  }

  std::set<int64_t> axis_set(axis_vec.begin(), axis_vec.end());
  return axis_set;
}

bool HaveReduceInPredecessors(const AnfNodePtr &node) {
  std::stack<AnfNodePtr> st;
  st.push(node);
  while (!st.empty()) {
    auto n = st.top();
    st.pop();

    if (n != node) {
      if (!n->isa<CNode>()) {
        continue;
      }
      if (IsPrimitiveCNode(n, prim::kPrimReduceSum)) {
        return true;
      }
    }

    auto n_inputs = n->cast<CNodePtr>()->inputs();
    (void)std::for_each(n_inputs.cbegin() + 1, n_inputs.cend(), [&st](const AnfNodePtr &n) -> void { st.push(n); });
  }

  return false;
}
}  // namespace

std::shared_ptr<AtomicAddChecker> AtomicAddChecker::Init() {
  auto processor = kernel::GetProcessorFromContext();
  if (processor == kernel::Processor::AICORE) {
    return std::make_shared<AtomicAddCheckerAscend>();
  } else if (processor == kernel::Processor::CUDA) {
    return std::make_shared<AtomicAddCheckerGPU>();
  }
  return nullptr;
}

bool AtomicAddChecker::FindCandidate(const AnfNodePtr &anf_node) {
  atomic_add_infos_.clear();
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  auto CheckSuitableTarget = [&mng_sub](const InplaceAssignerInfo &atomic_add_info) {
    // Target type should not fuse any other ops in out direction, which means it should be in output list.
    return mng_sub->node_users()[atomic_add_info.op_node].size() <= 1;
  };

  auto real_return_node = sub_graph->get_return()->input(kFirstDataInputIndex);
  InplaceAssignerInfo atomic_add_info;
  if (IsPrimitiveCNode(real_return_node, prim::kPrimMakeTuple)) {
    const auto &inputs = real_return_node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsPrimitiveCNode(inputs[i], target_type_)) {
        atomic_add_info.op_node = inputs[i]->cast<CNodePtr>();
        atomic_add_info.real_output_index = i - 1;
        atomic_add_info.real_output_num = inputs.size() - 1;
        // Target type should not fuse any other ops in out direction, which means it should be in output list.
        if (!CheckSuitableTarget(atomic_add_info)) {
          continue;
        }
        atomic_add_infos_.push_back(atomic_add_info);
      }
    }
  } else if (IsPrimitiveCNode(real_return_node, target_type_)) {
    atomic_add_info.op_node = real_return_node->cast<CNodePtr>();
    atomic_add_info.real_output_num = 1;
    if (CheckSuitableTarget(atomic_add_info)) {
      atomic_add_infos_.push_back(atomic_add_info);
    }
  } else {
    return false;
  }

  return !atomic_add_infos_.empty();
}

bool AtomicAddChecker::CanActivateAtomicAdd(const AnfNodePtr &anf_node) {
  // Rules to activate atomic add:
  // 1. Find only one ReduceSum inside sub-graph, and it should not fuse any other ops in out direction,
  //    which mean it should be in output list.
  // 2. The reduce axis and reduce number should meet condition:
  //    (GPU) all-reduce or reduce-x when fuse number is greater than or equal to 1024, or reduce-y.
  //    (Ascend) The first valid axis of the input data is the reduce axis or the non-reduce axis
  //    cannot make full use of multi-core.
  // 3. No other ReduceSum as output ReduceSum's predecessors (reduce compile limitation).

  // Rule 1.
  if (!FindCandidate(anf_node) || atomic_add_infos_.size() > 1) {
    return false;
  }

  // Rule 2.
  if (!SuitableForAtomicAdd(atomic_add_infos_[0].op_node)) {
    return false;
  }

  // Rule 3.
  return !HaveReduceInPredecessors(atomic_add_infos_[0].op_node);
}

bool AtomicAddChecker::Check(const AnfNodePtr &node) {
  return (common::AnfAlgo::IsGraphKernel(node) && CanActivateAtomicAdd(node));
}

bool AtomicAddCheckerGPU::SuitableForAtomicAdd(const AnfNodePtr &node) {
  auto input = node->cast<CNodePtr>()->input(kFirstDataInputIndex);
  auto src_shape_vec = GetShape(input);
  std::set<int64_t> axis_set = GetUniqReduceAxes(node);

  // For reduce whose last dim is reduced (including all-reduce),
  // it is suitable for atomic add only the reduce num is greater than or equal to 1024.
  if (axis_set.count(src_shape_vec.size() - 1) != 0) {
    size_t reduce_size = std::accumulate(
      axis_set.begin(), axis_set.end(), LongToSize(1),
      [&src_shape_vec](size_t size, int64_t axis) { return size * LongToSize(src_shape_vec[LongToSize(axis)]); });
    return reduce_size >= 1024;
  }

  // For reduce whose last dim is not reduced, always true.
  return true;
}

bool AtomicAddCheckerAscend::SuitableForAtomicAdd(const AnfNodePtr &node) {
  auto input = node->cast<CNodePtr>()->input(kFirstDataInputIndex);

  // Atomic addition is enabled only when the data type is fp32
  auto type = AnfAlgo::GetOutputDeviceDataType(input, 0);
  if (type != kNumberTypeFloat32) {
    return false;
  }

  // If the first valid axis of the input data is the reduce axis, enable atomic addition
  auto src_shape_vec = GetDeviceShape(input);
  std::set<int64_t> reduce_axis_set = GetUniqReduceAxes(node, true);
  auto start_with_reduce = false;
  for (size_t i = 0; i < src_shape_vec.size(); ++i) {
    auto dim = src_shape_vec[i];
    if (dim != 1) {
      if (reduce_axis_set.count(i) != 0) {
        start_with_reduce = true;
      }
      break;
    }
  }
  if (start_with_reduce) {
    return true;
  }

  // If the non-reduce axis cannot make full use of multi-core, enable atomic addition
  constexpr auto processor_core_num = 32LL;
  auto start_non_reduce_dim = 1LL;
  for (size_t i = 0; i < src_shape_vec.size(); ++i) {
    auto dim = src_shape_vec[i];
    if (reduce_axis_set.count(i) != 0) {
      break;
    }
    start_non_reduce_dim = start_non_reduce_dim * dim;
  }
  if (start_non_reduce_dim < processor_core_num) {
    return true;
  }

  return false;
}

void AtomicCleanInserter::InsertAtomicClean(const FuncGraphPtr &main_graph, const AnfNodePtr &anf_node,
                                            const std::vector<InplaceAssignerInfo> &atomic_add_infos,
                                            const FuncGraphManagerPtr &mng) {
  auto origin_composite_node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_composite_node);

  // Create broadcast node.
  std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> info_and_inplace_assignee_addr;
  for (auto atomic_add_info : atomic_add_infos) {
    auto out_type = GetType(atomic_add_info.op_node)->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(out_type);
    auto broadcast_to_node = CreateCleanCompositeNode(atomic_add_info, main_graph, out_type->element()->type_id());
    (void)info_and_inplace_assignee_addr.emplace_back(atomic_add_info, broadcast_to_node);
  }

  // Insert extra input(broadcast node output) to composite node, and make ReduceSum inplace-assign to it.
  ProcessOriginCNode(origin_composite_node, info_and_inplace_assignee_addr);

  // Insert Depend before origin ReduceSum's user to keep execution order.
  ProcessOriginCNodeUser(main_graph, origin_composite_node, info_and_inplace_assignee_addr, mng);
  std::stringstream ss;
  ss << "Target node: " << origin_composite_node->fullname_with_scope() << ", clean nodes: ";
  for (auto iter : info_and_inplace_assignee_addr) {
    ss << iter.second->fullname_with_scope() << ", ";
  }

  MS_LOG(INFO) << ss.str();
}

bool AtomicCleanInserter::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  std::shared_ptr<AtomicAddChecker> atomic_add_checker = AtomicAddChecker::Init();
  if (atomic_add_checker == nullptr) {
    return changed;
  }

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    if (!atomic_add_checker->Check(node)) {
      continue;
    }
    auto atomic_add_infos = atomic_add_checker->GetAtomicAddInfo();
    InsertAtomicClean(kernel_graph, node, atomic_add_infos, mng);
    changed = true;
  }

  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace mindspore::graphkernel
