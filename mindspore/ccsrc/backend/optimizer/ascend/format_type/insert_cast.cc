/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/format_type/insert_cast.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/optimizer/common/helper.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr InsertCastForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       const std::vector<bool> &need_insert_cast) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs;
  AbstractBasePtrList abstract_list;
  make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  size_t out_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < out_num; ++output_idx) {
    AnfNodePtr replace_node = nullptr;
    const auto origin_shape = AnfAlgo::GetOutputInferShape(cnode, output_idx);
    const auto infer_type = AnfAlgo::GetOutputInferDataType(cnode, output_idx);
    auto idx = NewValueNode(SizeToLong(output_idx));
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(output_idx);
    idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
    auto getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
    AnfAlgo::SetOutputInferTypeAndShape({infer_type}, {origin_shape}, getitem.get());
    if (need_insert_cast[output_idx]) {
      const auto dev_fmt = AnfAlgo::GetOutputFormat(cnode, output_idx);
      TypeId origin_type(kTypeUnknown);
      if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
        origin_type = AnfAlgo::GetCNodeOutputPrecision(cnode);
      }
      origin_type = origin_type == kTypeUnknown ? infer_type : origin_type;
      const auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_idx);
      if (origin_type != device_type) {
        replace_node = AddCastOpNodeToGraph(func_graph, getitem, dev_fmt, device_type, origin_type, origin_shape,
                                            infer_type, AnfAlgo::GetOutputReshapeType(getitem, 0));
        MS_EXCEPTION_IF_NULL(replace_node);
        replace_node->set_scope(cnode->scope());
        AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
        if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, output_idx)) {
          kernel_graph->ReplaceInternalOutput(cnode, replace_node, output_idx, 0);
        }
      } else {
        replace_node = getitem;
      }
    } else {
      replace_node = getitem;
    }
    abstract_list.emplace_back(replace_node->abstract());
    make_tuple_inputs.push_back(replace_node);
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple;
}  // namespace

AnfNodePtr InsertCastForOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                               const std::vector<bool> &need_insert_cast) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) == 0) {
    return cnode;
  }
  MS_EXCEPTION_IF_NULL(cnode->Type());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  // Single output
  if (!cnode->Type()->isa<Tuple>()) {
    if (!need_insert_cast[0]) {
      return cnode;
    }

    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, 0);
    std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(cnode, 0);
    const auto infer_type = AnfAlgo::GetOutputInferDataType(cnode, 0);
    TypeId origin_type(kTypeUnknown);
    if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      origin_type = AnfAlgo::GetCNodeOutputPrecision(cnode);
    }
    origin_type = origin_type == kTypeUnknown ? infer_type : origin_type;
    const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, 0);
    AnfNodePtr replace_node = cnode;
    if (origin_type != device_type) {
      replace_node = AddCastOpNodeToGraph(func_graph, cnode, dev_fmt, device_type, origin_type, origin_shape,
                                          infer_type, AnfAlgo::GetOutputReshapeType(cnode, 0));
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
      if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, 0)) {
        kernel_graph->ReplaceInternalOutput(cnode, replace_node);
      }
    }
    return replace_node;
  }
  // Multiple output
  return InsertCastForMultipleOutput(func_graph, cnode, need_insert_cast);
}

AnfNodePtr ProcessGraphKernelOp(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  // insert cast for ops in graph kernel.
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(sub_graph);
  auto mng = sub_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  std::vector<AnfNodePtr> todo;
  kernel::GetValidKernelNodes(sub_graph, &todo);
  auto outputs = AnfAlgo::GetAllOutput(sub_graph->output(), {prim::kPrimTupleGetItem});
  std::vector<std::pair<AnfNodePtr, size_t>> graph_rets;
  for (auto &output : outputs) {
    size_t index = 0;
    if (IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      ValuePtr tuple_index_value = GetValueNode(output->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      MS_EXCEPTION_IF_NULL(tuple_index_value);
      if (!tuple_index_value->isa<Int64Imm>()) {
        MS_LOG(EXCEPTION) << "The index of tuple getitem is not int64";
      }
      index = tuple_index_value->cast<Int64ImmPtr>()->value();
    }
    graph_rets.emplace_back(std::pair<AnfNodePtr, size_t>(output, index));
  }
  for (auto &t : todo) {
    AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), t);
    // process input
    CNodePtr t_cnode = t->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(t_cnode);
    auto t_new_node = InsertCastForInput(sub_graph, t_cnode);
    AnfNodePtr t_new_node_1 = nullptr;
    std::vector<bool> need_insert_cast(AnfAlgo::GetOutputTensorNum(t), true);
    // process output
    auto iter = std::find_if(graph_rets.begin(), graph_rets.end(),
                             [&t](const std::pair<AnfNodePtr, size_t> &ret) { return ret.first == t; });
    if (iter != graph_rets.end()) {
      auto t_fix_output_type = AnfAlgo::GetCNodeOutputPrecision(t);
      auto t_output_type = AnfAlgo::GetOutputDeviceDataType(t, iter->second);
      auto graph_output_type = AnfAlgo::GetOutputDeviceDataType(node, iter - graph_rets.begin());
      if (t_fix_output_type == kTypeUnknown && t_output_type == graph_output_type) {
        need_insert_cast[iter->second] = false;
      } else if (t_fix_output_type == t_output_type && t_output_type == graph_output_type) {
        need_insert_cast[iter->second] = false;
      }
      t_new_node_1 = InsertCastForOutput(sub_graph, t_new_node, need_insert_cast);
    } else {
      t_new_node_1 = InsertCastForOutput(sub_graph, t_new_node, need_insert_cast);
    }

    if (t_new_node_1 != nullptr && t_new_node_1 != t) {
      (void)mng->Replace(t, t_new_node_1);
    }
  }

  // insert cast for graph kernel.
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = InsertCastForInput(func_graph, cnode);
  // process output
  return InsertCastForOutput(func_graph, new_node, std::vector<bool>(AnfAlgo::GetOutputTensorNum(new_node), true));
}
}  // namespace

const BaseRef InsertCast::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertCast::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }

  if (AnfAlgo::IsGraphKernel(node)) {
    return ProcessGraphKernelOp(func_graph, node);
  }
  // insert cast for single op.
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = InsertCastForInput(func_graph, cnode);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(node)) {
    kernel_graph->ReplaceInternalOutput(node, new_node);
  }
  // process output
  return InsertCastForOutput(func_graph, new_node, std::vector<bool>(AnfAlgo::GetOutputTensorNum(new_node), true));
}
}  // namespace opt
}  // namespace mindspore
