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

#include "common/graph_kernel/add_atomic_clean.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "kernel/kernel.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/session/kernel_graph.h"
#include "debug/anf_ir_dump.h"
#include "kernel/common_utils.h"

namespace mindspore::graphkernel {
namespace {
auto constexpr NUMBER_COND_FOR_FILTER_INPLACE = 2;
std::set<int64_t> GetUniqReduceAxes(const AnfNodePtr &node, bool is_ascend = false) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceSum)) {
    MS_LOG(EXCEPTION) << "Expect ReduceSum node, but got " << AnfAlgo::GetCNodeName(node);
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
    for (size_t i = 0; i < src_shape_vec.size(); ++i) {
      (void)axis_vec.emplace_back(i);
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

inline int64_t CalNewIndex(int64_t old_index, const std::set<int64_t> &reduce_indexs) {
  int64_t count =
    std::count_if(reduce_indexs.begin(), reduce_indexs.end(), [old_index](int i) { return i < old_index; });
  return old_index - count;
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
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  auto CheckSuitableTarget = [&mng_sub](const AtomicAddInfo &atomic_add_info) {
    // Target type should not fuse any other ops in out direction, which means it should be in output list.
    return mng_sub->node_users()[atomic_add_info.atomic_add_node].size() <= 1;
  };

  auto real_return_node = sub_graph->get_return()->input(kFirstDataInputIndex);
  AtomicAddInfo atomic_add_info;
  if (IsPrimitiveCNode(real_return_node, prim::kPrimMakeTuple)) {
    const auto &inputs = real_return_node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsPrimitiveCNode(inputs[i], target_type_)) {
        atomic_add_info.atomic_add_node = inputs[i]->cast<CNodePtr>();
        atomic_add_info.reduce_real_output_index = i - 1;
        atomic_add_info.real_output_num = inputs.size() - 1;
        // Target type should not fuse any other ops in out direction, which means it should be in output list.
        if (!CheckSuitableTarget(atomic_add_info)) {
          continue;
        }
        atomic_add_infos_.push_back(atomic_add_info);
      }
    }
  } else if (IsPrimitiveCNode(real_return_node, target_type_)) {
    atomic_add_info.atomic_add_node = real_return_node->cast<CNodePtr>();
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
  if (!SuitableForAtomicAdd(atomic_add_infos_[0].atomic_add_node)) {
    return false;
  }

  // Rule 3.
  return !HaveReduceInPredecessors(atomic_add_infos_[0].atomic_add_node);
}

bool AtomicAddChecker::Check(const AnfNodePtr &node) {
  return (AnfAlgo::IsGraphKernel(node) && CanActivateAtomicAdd(node));
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
      if (reduce_axis_set.count(i)) {
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
    if (reduce_axis_set.count(i)) {
      break;
    }
    start_non_reduce_dim = start_non_reduce_dim * dim;
  }
  if (start_non_reduce_dim < processor_core_num) {
    return true;
  }

  return false;
}

void AtomicCleanInsertter::CorrectKernelBuildInfo(
  const AnfNodePtr &composite_node, const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &clean_infos) {
  // Change kernel build info.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_outputs_format = origin_kernel_build_info->GetAllOutputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();
  auto origin_outputs_type = origin_kernel_build_info->GetAllOutputDeviceTypes();
  auto origin_processor = origin_kernel_build_info->processor();

  std::vector<std::string> &new_inputs_format = origin_inputs_format;
  std::vector<TypeId> &new_inputs_type = origin_inputs_type;
  std::vector<std::string> new_outputs_format;
  std::vector<TypeId> new_outputs_type;

  std::set<size_t> reduce_real_indices;
  for (auto &info : clean_infos) {
    (void)reduce_real_indices.insert(info.first.reduce_real_output_index);
  }

  if (clean_infos[0].first.real_output_num == reduce_real_indices.size()) {
    new_outputs_format.push_back(origin_outputs_format[0]);
    new_outputs_type.push_back(origin_outputs_type[0]);
  } else {
    for (size_t i = 0; i < origin_outputs_format.size(); ++i) {
      if (reduce_real_indices.count(i) > 0) {
        continue;
      }
      new_outputs_format.push_back(origin_outputs_format[i]);
      new_outputs_type.push_back(origin_outputs_type[i]);
    }
  }

  for (const auto &clean_info : clean_infos) {
    auto &new_input = clean_info.second;
    auto kernel_with_index = AnfAlgo::VisitKernel(new_input, 0);
    new_inputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
    new_inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder new_info_builder;
  new_info_builder.SetInputsFormat(new_inputs_format);
  new_info_builder.SetInputsDeviceType(new_inputs_type);
  new_info_builder.SetOutputsFormat(new_outputs_format);
  new_info_builder.SetOutputsDeviceType(new_outputs_type);
  new_info_builder.SetProcessor(origin_processor);
  new_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  new_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto new_selected_info = new_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void AtomicCleanInsertter::CreateInplaceAssignNodeAndCorrectReturn(
  const FuncGraphPtr &sub_graph, const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &parameters_infos) {
  // Add inplaceassign
  AnfNodePtr inplace_out_node;
  bool fake_out = false;

  std::set<size_t> reduce_indices;
  for (auto &info : parameters_infos) {
    (void)reduce_indices.insert(info.first.reduce_real_output_index + 1);
  }
  size_t replace_index = 0;
  auto retrun_node = sub_graph->get_return()->input(kFirstDataInputIndex);
  if (!IsPrimitiveCNode(retrun_node, prim::kPrimMakeTuple) ||
      retrun_node->cast<CNodePtr>()->inputs().size() == parameters_infos.size() + 1) {
    fake_out = true;
    inplace_out_node = parameters_infos[0].first.atomic_add_node;
    replace_index = parameters_infos[0].first.reduce_real_output_index + 1;
  } else {
    const auto &outs = retrun_node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < outs.size(); ++i) {
      if (reduce_indices.count(i) == 0) {
        inplace_out_node = outs[i];
        replace_index = i;
        break;
      }
    }
  }

  for (const auto &[atomic_add_info, new_parameter] : parameters_infos) {
    auto inplace_assign_node = CreateCNode(
      {NewValueNode(prim::kPrimInplaceAssign), new_parameter, atomic_add_info.atomic_add_node, inplace_out_node},
      sub_graph,
      {.format = GetFormat(inplace_out_node), .shape = GetShape(inplace_out_node), .type = GetType(inplace_out_node)});
    SetNodeAttrSafely("fake_output", MakeValue(fake_out), inplace_assign_node);
    inplace_out_node = inplace_assign_node;
  }

  CNodePtr new_out_node;
  // If the real output number is less than or equal to two, it's no need to filter the inplace one out:
  // 1. Two real outputs. After inplacing, only one left and `Inplace` out will be that output.
  // 2. One real output. After inplacing, there is no output left, use fake one.
  if (parameters_infos[0].first.real_output_num > NUMBER_COND_FOR_FILTER_INPLACE) {
    std::vector<AnfNodePtr> output_args = {NewValueNode(prim::kPrimMakeTuple)};
    const auto &outs = retrun_node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < outs.size(); ++i) {
      if (reduce_indices.count(i) > 0) {
        continue;
      } else if (i == replace_index) {
        output_args.push_back(inplace_out_node);
      } else {
        output_args.push_back(outs[i]);
      }
    }
    // Set output for AnfGraph
    new_out_node = sub_graph->NewCNode(output_args);
  } else {
    CNodePtr out_cnode = inplace_out_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    new_out_node = out_cnode;
  }
  sub_graph->set_output(new_out_node);
}

void AtomicCleanInsertter::CorrectAbstract(
  const AnfNodePtr &composite_node, const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &process_infos) const {
  // If there is only one output, it should be a fake output with the same abstract with origin output.
  if (process_infos[0].first.real_output_num <= 1) {
    return;
  }

  std::set<size_t> reduce_real_indices;
  for (auto &info : process_infos) {
    (void)reduce_real_indices.insert(info.first.reduce_real_output_index);
  }

  // Change abstract.
  auto origin_out_spec = composite_node->abstract()->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(origin_out_spec);
  const auto &origin_out_specs = origin_out_spec->elements();
  AbstractBasePtrList new_out_specs;
  for (size_t i = 0; i < origin_out_specs.size(); ++i) {
    if (reduce_real_indices.count(i) == 0) {
      new_out_specs.push_back(origin_out_specs[i]);
    }
  }

  // If empty, there will be a fake out, so use the first target reduce information.
  if (new_out_specs.empty()) {
    new_out_specs.push_back(origin_out_specs[process_infos[0].first.reduce_real_output_index]);
  }
  composite_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_out_specs));
}

void AtomicCleanInsertter::ProcessOriginCNode(
  const AnfNodePtr &composite_node,
  const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes) {
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // Add input
  std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> parameters_infos;
  for (const auto &[atomic_add_info, new_input] : info_and_broadcast_to_nodes) {
    // Add atomic attribute to reducesum node.
    SetNodeAttrSafely("enable_atomic_add", MakeValue(true), atomic_add_info.atomic_add_node);
    // add parameter
    auto parameter = sub_graph->add_parameter();
    parameter->set_abstract(new_input->abstract());
    parameter->set_kernel_info(new_input->kernel_info_ptr());
    (void)parameters_infos.emplace_back(atomic_add_info, parameter);
  }

  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  (void)std::transform(info_and_broadcast_to_nodes.cbegin(), info_and_broadcast_to_nodes.cend(),
                       std::back_inserter(inputs),
                       [](const std::pair<AtomicAddInfo, AnfNodePtr> &pair_item) { return pair_item.second; });
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  CreateInplaceAssignNodeAndCorrectReturn(sub_graph, parameters_infos);

  CorrectAbstract(composite_node, info_and_broadcast_to_nodes);
  CorrectKernelBuildInfo(composite_node, info_and_broadcast_to_nodes);

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name = GkUtils::ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "atomic_add");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to atomic add graph " << new_graph_name;
}

CNodePtr AtomicCleanInsertter::InsertUpdateState(const KernelGraphPtr &main_graph,
                                                 const CNodePtr &composite_node) const {
  // Insert update_state_node, need mount a monad node.
  auto u = NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  AnfNodePtrList update_state_inputs = {NewValueNode(prim::kPrimUpdateState), u, composite_node};
  auto update_state_cnode = main_graph->NewCNode(update_state_inputs);
  update_state_cnode->set_abstract(kUMonad->ToAbstract());
  main_graph->AddNode(update_state_cnode);
  return update_state_cnode;
}

CNodePtr AtomicCleanInsertter::CreateAtomicCleanCompositeNode(const AtomicAddInfo &atomic_add_info,
                                                              const KernelGraphPtr &main_graph, TypeId dst_type) {
  std::set<TypeId> data_support = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};

  if (!std::any_of(data_support.cbegin(), data_support.cend(), [&dst_type](TypeId type) { return dst_type == type; })) {
    MS_LOG(EXCEPTION) << "For AtomicAdd, the data type: " << TypeIdToString(dst_type, true)
                      << " is not in supported list: [float16, float32, float64].";
  }

  // Create zero value which will be broadcast to target shape.
  auto format = GetFormat(atomic_add_info.atomic_add_node);
  auto dtype = (dst_type == kNumberTypeFloat16) ? kNumberTypeFloat32 : dst_type;
  ValueNodePtr value_node;
  if (dtype == kNumberTypeFloat32) {
    value_node = CreateScalarTensorValueNode<float>({.format = format, .shape = {1}, .type = TypeIdToType(dtype)},
                                                    static_cast<float>(0), sizeof(float));
  } else {
    value_node = CreateScalarTensorValueNode<double>({.format = format, .shape = {1}, .type = TypeIdToType(dtype)},
                                                     static_cast<double>(0), sizeof(double));
  }

  // Create composite op's sub-graph.
  auto new_sub_graph = std::make_shared<FuncGraph>();

  AnfNodePtr broadcast_input_node;
  if (dst_type == kNumberTypeFloat16) {
    AnfNodePtrList cast_inputs = {NewValueNode(prim::kPrimCast), value_node};
    auto cast_node_inner =
      CreateCNode(cast_inputs, new_sub_graph, {.format = format, .shape = {1}, .type = TypeIdToType(dst_type)});
    SetNodeAttrSafely("dst_type", MakeValue("float32"), cast_node_inner);
    broadcast_input_node = cast_node_inner;
  } else {
    broadcast_input_node = value_node;
  }

  // Create broadcast basic op.
  auto dst_shape_vec = GetShape(atomic_add_info.atomic_add_node);
  AnfNodePtrList atomic_clean_inputs = {NewValueNode(prim::kPrimBroadcastTo), broadcast_input_node};
  auto broadcast_to_node_inner =
    CreateCNode(atomic_clean_inputs, new_sub_graph,
                {.format = format, .shape = dst_shape_vec, .type = GetType(atomic_add_info.atomic_add_node)});
  SetNodeAttrSafely("shape", MakeValue(GetDeviceShape(atomic_add_info.atomic_add_node)), broadcast_to_node_inner);

  // Makeup sub-graph.
  new_sub_graph->set_output(broadcast_to_node_inner);
  auto broadcast_to_composite_node = main_graph->NewCNode({NewValueNode(new_sub_graph)});
  broadcast_to_composite_node->set_abstract(broadcast_to_node_inner->abstract());
  SetNewKernelInfo(broadcast_to_composite_node, new_sub_graph, {}, {broadcast_to_node_inner});
  auto graph_attr = GkUtils::ExtractGraphKernelName(TopoSort(new_sub_graph->get_return()), "", "atomic_clean");
  new_sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(graph_attr));
  new_sub_graph->set_attr("composite_type", MakeValue("atomic_clean"));

  return broadcast_to_composite_node;
}

std::vector<std::tuple<AnfNodePtr, int, AnfNodePtr>> AtomicCleanInsertter::FindOriginCNodeUsers(
  const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
  const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes, const FuncGraphManagerPtr &mng,
  bool correct_index) const {
  std::vector<std::tuple<AnfNodePtr, int, AnfNodePtr>> reduce_user_nodes;

  std::set<int64_t> real_indices;
  std::map<size_t, AnfNodePtr> real_indices_and_clean_node;
  for (auto &[info, clean] : info_and_broadcast_to_nodes) {
    (void)real_indices_and_clean_node.emplace(info.reduce_real_output_index, clean);
    (void)real_indices.insert(SizeToLong(info.reduce_real_output_index));
  }

  if (info_and_broadcast_to_nodes[0].first.real_output_num <= 1) {
    auto users = mng->node_users()[composite_node];
    for (const auto &[user, index] : users) {
      (void)reduce_user_nodes.emplace_back(user, index, info_and_broadcast_to_nodes[0].second);
    }
  } else {
    std::vector<std::tuple<AnfNodePtr, AnfNodePtr>> getitem_user_nodes;
    auto users = mng->node_users()[composite_node];
    for (const auto &node_index : users) {
      const auto &user_node = node_index.first;
      if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto get_item_cnode = user_node->cast<CNodePtr>();
      auto value_input = get_item_cnode->input(kInputNodeOutputIndexInTupleGetItem);
      MS_EXCEPTION_IF_NULL(value_input);
      auto value_node = value_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto item_idx = GetValue<int64_t>(value_node->value());
      auto iter = real_indices_and_clean_node.find(LongToSize(item_idx));
      if (iter != real_indices_and_clean_node.end()) {
        (void)getitem_user_nodes.emplace_back(node_index.first, iter->second);
      } else if (correct_index) {
        // Recorrect other getitem index.
        int64_t new_item_idx = CalNewIndex(item_idx, real_indices);
        AnfNodePtrList new_inputs = {NewValueNode(prim::kPrimTupleGetItem), composite_node, NewValueNode(new_item_idx)};
        auto new_out = main_graph->NewCNode(new_inputs);
        new_out->set_abstract(get_item_cnode->abstract());
        for (const auto &[user, index] : mng->node_users()[get_item_cnode]) {
          auto user_cnode = user->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(user_cnode);
          user_cnode->set_input(IntToSize(index), new_out);
        }
      }
    }
    for (auto &[getitem_node, broadcast_to_node] : getitem_user_nodes) {
      // Directory to find real user.
      auto real_users = mng->node_users()[getitem_node];
      (void)std::transform(real_users.cbegin(), real_users.cend(), std::back_inserter(reduce_user_nodes),
                           [&broadcast_to_node](const std::pair<AnfNodePtr, int> &pair) {
                             return std::make_tuple(pair.first, pair.second, broadcast_to_node);
                           });
    }
  }

  return reduce_user_nodes;
}

void AtomicCleanInsertter::ProcessOriginCNodeUser(
  const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
  const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes,
  const AnfNodePtr &update_state_node, const FuncGraphManagerPtr &mng) {
  // 1. find users, change getitem index if needed.
  std::vector<std::tuple<AnfNodePtr, int, AnfNodePtr>> reduce_user_nodes =
    FindOriginCNodeUsers(main_graph, composite_node, info_and_broadcast_to_nodes, mng, true);
  for (const auto &[user_node, index, broadcast_to_node] : reduce_user_nodes) {
    // 2. Make sure modified composite node running first, So firstly, create load_node, then add edge to connect
    // update_state_node, broadcat_node and load_node to keep order.
    AnfNodePtrList load_inputs = {NewValueNode(prim::kPrimLoad), broadcast_to_node, update_state_node};
    auto load_node = main_graph->NewCNode(load_inputs);
    load_node->set_abstract(broadcast_to_node->abstract());
    main_graph->AddNode(load_node);
    auto user_cnode = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    user_cnode->set_input(IntToSize(index), load_node);
  }
}

void AtomicCleanInsertter::InsertAtomicClean(const KernelGraphPtr &main_graph, const AnfNodePtr &anf_node,
                                             const std::vector<AtomicAddInfo> &atomic_add_infos,
                                             const FuncGraphManagerPtr &mng) {
  auto origin_composite_node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_composite_node);

  // Create broadcst node.
  std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> info_and_broadcast_to_nodes;
  for (auto atomic_add_info : atomic_add_infos) {
    auto out_type = GetType(atomic_add_info.atomic_add_node)->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(out_type);
    auto broadcast_to_node =
      CreateAtomicCleanCompositeNode(atomic_add_info, main_graph, out_type->element()->type_id());
    (void)info_and_broadcast_to_nodes.emplace_back(atomic_add_info, broadcast_to_node);
  }

  // Insert extra input(broadcast node output) to composite node, and make Reducesum inplaceassign to it.
  // Note: if it's single output, this will increase total memory because of a fake out.
  ProcessOriginCNode(origin_composite_node, info_and_broadcast_to_nodes);

  // Insert update_state_node to keep execution order.
  auto update_state_node = InsertUpdateState(main_graph, origin_composite_node);

  // Replace origin ReduceSum's user with atomic clean output
  ProcessOriginCNodeUser(main_graph, origin_composite_node, info_and_broadcast_to_nodes, update_state_node, mng);
  std::stringstream ss;
  ss << "Target node: " << origin_composite_node->fullname_with_scope() << ", clean nodes: ";
  for (auto iter : info_and_broadcast_to_nodes) {
    ss << iter.second->fullname_with_scope() << ", ";
  }

  MS_LOG(INFO) << ss.str();
}

bool AtomicCleanInsertter::Run(const FuncGraphPtr &func_graph) {
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
