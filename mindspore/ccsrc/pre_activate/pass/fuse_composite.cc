
/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/pass/fuse_composite.h"

#include <memory>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
#include <vector>

#include "operator/ops.h"
#include "utils/utils.h"
#include "utils/graph_utils.h"
#include "pre_activate/common/helper.h"
#include "session/anf_runtime_algorithm.h"
#include "vm/segment_runner.h"
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
std::vector<PrimitivePtr> get_fusable_basic_ops(bool is_before_kernel_select) {
  std::vector<PrimitivePtr> fusable_basic_ops = {
    prim::kPrimAddN,       prim::kPrimTensorAdd,  prim::kPrimMul,      prim::kPrimSub, prim::kPrimMaximum,
    prim::kPrimMinimum,    prim::kPrimNeg,        prim::kPrimRealDiv,  prim::kPrimPow, prim::kPrimSqrt,
    prim::kPrimReciprocal, prim::kPrimExpandDims, prim::kPrimLessEqual};
  if (!is_before_kernel_select) {
    fusable_basic_ops.push_back(prim::kPrimCast);
  }
  return fusable_basic_ops;
}

std::vector<PrimitivePtr> get_fusable_basic_ops_with_reduce(bool is_before_kernel_select) {
  std::vector<PrimitivePtr> fusable_basic_ops_with_reduce;
  if (!is_before_kernel_select) {
    fusable_basic_ops_with_reduce.push_back(prim::kPrimCast);
  }
  return fusable_basic_ops_with_reduce;
}

std::vector<PrimitivePtr> get_reduce_ops() {
  std::vector<PrimitivePtr> reduce_ops = {prim::kPrimReduceSum, prim::kPrimReduceMean, prim::kPrimReduceMin,
                                          prim::kPrimReduceMax, prim::kPrimReduceAll};
  return reduce_ops;
}

void GetCompositeInfo(const FuncGraphPtr fg, CompositeInfo *info) {
  MS_EXCEPTION_IF_NULL(fg);
  auto reduce_ops = get_reduce_ops();
  const auto &nodes = fg->nodes();
  info->op_type = ELEWISE;
  info->cal_step = -1;
  info->reduce_op_num = 0;
  for (auto node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    info->cal_step++;
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr) {
      bool is_reudce = std::any_of(reduce_ops.begin(), reduce_ops.end(), [&prim](const PrimitivePtr &op) {
        return op->hash() == prim->hash() && op->name() == prim->name();
      });
      if (is_reudce) {
        info->op_type = REDUCE;
        info->reduce_op_num++;
      }
    }
  }
}

bool IsFuse(const CompositeInfo &info, const AnfNodePtr &node) {
  auto fusable_basic_ops = get_fusable_basic_ops(info.is_before_kernel_select);
  auto fusable_basic_ops_with_reduce = get_fusable_basic_ops_with_reduce(info.is_before_kernel_select);
  bool is_fusable = false;
  if (info.op_type == REDUCE &&
      (info.cal_step >= MAX_REDUCE_OP_FUSION_CAL_STEP || info.reduce_op_num >= MAX_REDUCE_OP_FUSION_REDUCE_NUM)) {
    is_fusable = std::any_of(fusable_basic_ops_with_reduce.begin(), fusable_basic_ops_with_reduce.end(),
                             [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  } else {
    is_fusable = std::any_of(fusable_basic_ops.begin(), fusable_basic_ops.end(),
                             [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  }

  return is_fusable;
}

IncludeType IncludeFusedBasicOpForward(const AnfNodePtr &cur_node, const CompositeInfo &info, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (!IsPrimitiveCNode(node)) {
    return EXCLUDE;
  }

  bool is_fusable = IsFuse(info, node);
  return is_fusable ? FOLLOW : EXCLUDE;
}

IncludeType IncludeFusedBasicOpBackward(const AnfNodePtr &cur_node, const CompositeInfo &info, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (AnfAlgo::IsCompositeKernel(node)) {
    auto cnode = node->cast<CNodePtr>();
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(kAnfPrimitiveIndex));
    auto fg_attr_val = fg->get_attr(FUNC_GRAPH_FLAG_COMPOSITE);
    MS_EXCEPTION_IF_NULL(fg_attr_val);
    auto fg_attr = GetValue<std::string>(fg_attr_val);
    if (fg_attr == kApplyMomentumOpName) {
      return FOLLOW;
    }
    return EXCLUDE;
  }
  if (!IsPrimitiveCNode(node)) {
    return EXCLUDE;
  }

  bool is_fusable = IsFuse(info, node);
  return is_fusable ? FOLLOW : EXCLUDE;
}

bool CheckCircle(const std::set<AnfNodePtr> &fused_op_set, const AnfNodePtr &check_node,
                 std::set<AnfNodePtr> *cached_unconnected_set) {
  if (!check_node->isa<CNode>() || AnfAlgo::IsCompositeKernel(check_node)) {
    return false;
  }

  auto cnode = check_node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  // there is a input not in fused_op_set, but the input depends on the fused_op_set
  bool has_circle = false;
  for (auto input : inputs) {
    if (input->isa<CNode>() && !fused_op_set.count(input)) {
      std::set<AnfNodePtr> done;
      std::vector<AnfNodePtr> todos = {input};
      while (!todos.empty()) {
        auto node = todos.back();
        todos.pop_back();
        if (done.count(node) || cached_unconnected_set->count(node)) {
          continue;
        }

        done.insert(node);
        if (fused_op_set.count(node)) {
          has_circle = true;
          break;
        }

        if (node->isa<CNode>()) {
          auto cnode_ptr = node->cast<CNodePtr>();
          for (auto it : cnode_ptr->inputs()) {
            if (it->isa<CNode>()) {
              todos.push_back(it);
            }
          }
        }
      }

      if (has_circle) {
        return true;
      }
      cached_unconnected_set->insert(done.begin(), done.end());
    }
  }

  return false;
}

bool IsMakeTupleOut(const AnfNodePtr &out, AnfNodePtrList *real_outs) {
  if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
    auto &inputs = out->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      real_outs->push_back(inputs[i]);
    }
    return true;
  }

  if (AnfAlgo::GetCNodeFuncGraphPtr(out) != nullptr) {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(out);
    auto fg_out = fg->output();
    if (IsPrimitiveCNode(fg_out, prim::kPrimMakeTuple)) {
      auto inputs = fg_out->cast<CNodePtr>()->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        real_outs->push_back(inputs[i]);
      }
      return true;
    }
  }
  return false;
}

std::vector<AnfNodePtr> RemoveCircle(const std::vector<AnfNodePtr> &fused_op, bool is_backward) {
  std::set<AnfNodePtr> cached_unconnected_set;
  std::set<AnfNodePtr> fused_op_set(fused_op.begin(), fused_op.end());
  auto include = [&fused_op_set](const AnfNodePtr &node) {
    if (fused_op_set.count(node)) {
      return FOLLOW;
    }
    return EXCLUDE;
  };
  for (auto iter = fused_op.rbegin(); iter != fused_op.rend(); ++iter) {
    bool has_circle = CheckCircle(fused_op_set, *iter, &cached_unconnected_set);
    // delete the circle node and the node which depend on the circle node in fused op
    if (has_circle) {
      auto mng = (*iter)->func_graph()->manager();
      std::vector<AnfNodePtr> erase_nodes;
      if (is_backward) {
        erase_nodes = DeepUsersSearch(*iter, include, mng);
      } else {
        erase_nodes = DeepLinkedGraphSearch(*iter, include);
      }
      for (auto erase_node : erase_nodes) {
        fused_op_set.erase(erase_node);
      }
    }
  }

  std::vector<AnfNodePtr> res;
  for (auto node : fused_op) {
    if (fused_op_set.count(node)) {
      res.push_back(node);
    }
  }
  return res;
}

void TopoSortForNodeList(std::vector<AnfNodePtr> *lst) {
  if (lst->size() < 2) {
    return;
  }

  std::vector<AnfNodePtr> res;
  std::set<AnfNodePtr> node_sets(lst->begin(), lst->end());
  std::map<AnfNodePtr, std::set<AnfNodePtr>> ins;
  std::map<AnfNodePtr, std::set<AnfNodePtr>> outs;
  std::queue<AnfNodePtr> q;
  for (auto node : *lst) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto input : cnode->inputs()) {
      if (!node_sets.count(input)) {
        continue;
      }
      // out_degree
      outs[input].insert(node);
      // in_degree
      ins[node].insert(input);
    }
    if (!ins.count(node)) {
      ins[node] = {};
    }
  }

  for (auto p : ins) {
    if (p.second.size() == 0) {
      q.push(p.first);
    }
  }

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    res.push_back(node);
    if (!outs.count(node)) {
      continue;
    }
    for (auto out : outs[node]) {
      if (!ins.count(out)) {
        continue;
      }
      ins[out].erase(node);
      if (ins[out].size() == 0) {
        q.push(out);
      }
    }
  }

  lst->assign(res.begin(), res.end());
}

std::vector<AnfNodePtr> FindFuseCNodes(const CNodePtr &cnode, bool is_before_kernel_select) {
  auto func_graph = cnode->func_graph();
  auto composite_g = GetValueNode<FuncGraphPtr>(cnode->input(0));
  CompositeInfo info;
  info.is_before_kernel_select = is_before_kernel_select;
  GetCompositeInfo(composite_g, &info);
  auto mng = func_graph->manager();
  // Search fusable nodes according input direction.
  auto include_func_forward = std::bind(IncludeFusedBasicOpForward, cnode, info, std::placeholders::_1);
  auto used_nodes = DeepLinkedGraphSearch(cnode, include_func_forward);
  std::reverse(used_nodes.begin(), used_nodes.end());
  // Search fusable nodes according output direction.
  auto include_func_backward = std::bind(IncludeFusedBasicOpBackward, cnode, info, std::placeholders::_1);
  auto user_nodes = DeepUsersSearch(cnode, include_func_backward, mng);

  used_nodes.insert(used_nodes.end(), user_nodes.begin() + 1, user_nodes.end());
  if (used_nodes.size() > 1) {
    used_nodes = RemoveCircle(used_nodes);
  }
  TopoSortForNodeList(&used_nodes);
  return used_nodes;
}

AbstractBasePtr GetOutputAbstract(const AnfNodePtr &node, size_t output_idx) {
  auto out_spec = node->abstract();
  if (out_spec->isa<abstract::AbstractTuple>()) {
    return out_spec->cast<abstract::AbstractTuplePtr>()->elements()[output_idx];
  }
  return out_spec;
}

AnfNodePtr CreateNewFuseCNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const FuncGraphPtr &fg,
                              const AnfNodePtrList &inputs, const AnfNodePtrList &outputs,
                              bool is_before_kernel_select) {
  auto func_node = NewValueNode(fg);
  std::vector<AnfNodePtr> fn_inputs;
  fn_inputs.push_back(func_node);
  fn_inputs.insert(fn_inputs.end(), inputs.begin(), inputs.end());
  auto fuse_cnode = kernel_graph->NewCNode(fn_inputs);
  // Set output abstract
  if (outputs.size() > 1) {
    std::vector<AbstractBasePtr> out_specs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      out_specs.push_back(outputs[i]->abstract());
    }
    auto out_spec = std::make_shared<abstract::AbstractTuple>(out_specs);
    fuse_cnode->set_abstract(out_spec);
  } else {
    fuse_cnode->set_abstract(outputs[0]->abstract());
  }
  // Set parameter abstract.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
    auto input_abs = GetOutputAbstract(kernel_with_index.first, kernel_with_index.second);
    fg->parameters()[i]->set_abstract(input_abs);
    if (is_before_kernel_select) {
      fg->parameters()[i]->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
  }
  // Set kernel info.
  if (!is_before_kernel_select) {
    std::vector<std::string> graph_input_format;
    std::vector<TypeId> graph_input_type;
    std::vector<std::string> graph_output_format;
    std::vector<TypeId> graph_output_type;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      graph_input_format.push_back(input_format);
      auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      graph_input_type.push_back(input_type);
      auto input_abs = GetOutputAbstract(kernel_with_index.first, kernel_with_index.second);
      fg->parameters()[i]->set_abstract(input_abs);
    }
    auto new_outputs = outputs;
    if (outputs.size() == 1 && AnfAlgo::IsCompositeKernel(outputs[0])) {
      std::vector<AnfNodePtr> real_outs;
      if (IsMakeTupleOut(outputs[0], &real_outs)) {
        new_outputs = real_outs;
      }
    }
    for (size_t i = 0; i < new_outputs.size(); ++i) {
      auto kernel_with_index = AnfAlgo::VisitKernel(new_outputs[i], 0);
      auto output_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      graph_output_format.push_back(output_format);
      graph_output_type.push_back(output_type);
    }
    kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
    graph_info_builder.SetInputsFormat(graph_input_format);
    graph_info_builder.SetInputsDeviceType(graph_input_type);
    graph_info_builder.SetOutputsFormat(graph_output_format);
    graph_info_builder.SetOutputsDeviceType(graph_output_type);
    graph_info_builder.SetProcessor(kernel::Processor::AICORE);
    graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
    graph_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
    auto graph_selected_info = graph_info_builder.Build();
    AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, fuse_cnode.get());
  }
  return fuse_cnode;
}

void ReplaceNewFuseCNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // single out
  if (outputs.size() == 1) {
    mng->Replace(outputs[0], new_fuse_cnode);
    return;
  }

  std::vector<AnfNodePtr> fn_inputs;
  for (size_t out_idx = 0; out_idx < outputs.size(); out_idx++) {
    AnfNodePtrList real_outs;
    // not make tuple out, replace
    if (!IsMakeTupleOut(outputs[out_idx], &real_outs)) {
      fn_inputs.clear();
      fn_inputs.push_back(NewValueNode(prim::kPrimTupleGetItem));
      fn_inputs.push_back(new_fuse_cnode);
      fn_inputs.push_back(NewValueNode(MakeValue(SizeToInt(out_idx))));
      auto new_out = kernel_graph->NewCNode(fn_inputs);
      new_out->set_abstract(outputs[out_idx]->abstract());
      mng->Replace(outputs[out_idx], new_out);
      continue;
    }

    // the out is make tuple , modify the get_item node's value
    auto users = mng->node_users()[outputs[out_idx]];
    for (auto &user : users) {
      auto use_node = user.first;
      if (use_node->isa<CNode>() && (IsPrimitiveCNode(use_node, prim::kPrimTupleGetItem))) {
        auto get_item_cnode = use_node->cast<CNodePtr>();
        auto value_input = get_item_cnode->input(kInputNodeOutputIndexInTupleGetItem);
        MS_EXCEPTION_IF_NULL(value_input);
        auto value_node = value_input->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        int item_idx = GetValue<int>(value_node->value());
        int new_item_idx = SizeToInt(out_idx) + item_idx;
        fn_inputs.clear();
        fn_inputs.push_back(NewValueNode(prim::kPrimTupleGetItem));
        fn_inputs.push_back(new_fuse_cnode);
        fn_inputs.push_back(NewValueNode(new_item_idx));
        auto new_out = kernel_graph->NewCNode(fn_inputs);
        new_out->set_abstract(get_item_cnode->abstract());
        mng->Replace(get_item_cnode, new_out);
      }
    }
  }
}

AnfNodePtrList EliminateMakeTuple(FuncGraphPtr *fg, FuncGraphManagerPtr *mng) {
  AnfNodePtrList outs;
  auto out_node = (*fg)->output();
  if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
    std::vector<AnfNodePtr> output_args;
    auto out_cnode = out_node->cast<CNodePtr>();
    for (auto out : out_cnode->inputs()) {
      if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
        auto inputs = out->cast<CNodePtr>()->inputs();
        for (size_t i = 1; i < inputs.size(); ++i) {
          output_args.push_back(inputs[i]);
        }
      } else {
        output_args.push_back(out);
      }
    }
    if (output_args.size() != out_cnode->inputs().size()) {
      auto new_out = (*fg)->NewCNode(output_args);
      (*mng)->Replace(out_node, new_out);
    }

    for (size_t i = 1; i < output_args.size(); ++i) {
      outs.push_back(output_args[i]);
    }
    return outs;
  }

  outs.push_back(out_node);
  return outs;
}

AnfNodePtrList GetExpandOuts(const AnfNodePtrList &outs) {
  AnfNodePtrList res;
  if (outs.size() <= 1) {
    return outs;
  }

  for (auto out : outs) {
    AnfNodePtrList real_outs;
    if (IsMakeTupleOut(out, &real_outs)) {
      res.insert(res.end(), real_outs.begin(), real_outs.end());
      continue;
    }
    res.push_back(out);
  }
  return res;
}

void FuseComposite(const std::shared_ptr<session::KernelGraph> &kernel_graph, bool is_before_kernel_select) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  auto &todos = kernel_graph->execution_order();
  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto node = *iter;
    if (!AnfAlgo::IsCompositeKernel(node) || !kernel_graph->nodes().contains(node)) {
      continue;
    }

    auto origin_fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
    auto fg_attr = origin_fg->get_attr(FUNC_GRAPH_FLAG_COMPOSITE);
    if (fg_attr != nullptr) {
      auto fg_name = GetValue<std::string>(fg_attr);
      if (composite_black_list.count(fg_name) != 0) {
        continue;
      }
    }

    auto fuse_nodes = FindFuseCNodes(node, is_before_kernel_select);
    if (fuse_nodes.size() <= 1) {
      continue;
    }

    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(fuse_nodes);

    // Remove nest make tuple in outs
    auto expand_out = GetExpandOuts(outputs);
    auto fuse_new_node = CreateNewFuseCNode(kernel_graph, fg, inputs, expand_out, is_before_kernel_select);

    ReplaceNewFuseCNode(kernel_graph, fuse_new_node, outputs);

    // Inline origin composite
    auto cnodes = fg->GetOrderedCnodes();
    for (const auto &n : cnodes) {
      if (!AnfAlgo::IsCompositeKernel(n)) {
        continue;
      }
      auto composite_g = GetValueNode<FuncGraphPtr>(n->input(0));
      AnfNodePtrList ins;
      ins.insert(ins.end(), n->inputs().begin() + 1, n->inputs().end());
      auto out = InlineClone(composite_g, fg, ins, n->input(0)->scope());
      mng->Replace(n, out);
    }

    EliminateMakeTuple(&fg, &mng);
    // Set composite flag
    auto ori_fg = GetValueNode<FuncGraphPtr>(node->input(kAnfPrimitiveIndex));
    fg->set_attr(FUNC_GRAPH_FLAG_COMPOSITE, ori_fg->get_attr(FUNC_GRAPH_FLAG_COMPOSITE));
  }
}
}  // namespace opt
}  // namespace mindspore
