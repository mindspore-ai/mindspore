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
#include "plugin/device/ascend/optimizer/format_type/insert_cast.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "backend/common/optimizer/helper.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr InsertCastForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_cnode,
                                       const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto update_states = common::AnfAlgo::GetUpdateStateUsers(manager, orig_cnode);
  for (auto &update_state : update_states) {
    manager->SetEdge(update_state.first, update_state.second, cnode);
  }
  if (manager->node_users()[orig_cnode].empty()) {
    return cnode;
  }
  std::vector<AnfNodePtr> make_tuple_inputs;
  AbstractBasePtrList abstract_list;
  make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  size_t out_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < out_num; ++output_idx) {
    AnfNodePtr replace_node = nullptr;
    const auto origin_shape = AnfAlgo::GetOutputDetailShape(cnode, output_idx);
    const auto origin_type = common::AnfAlgo::GetOutputInferDataType(cnode, output_idx);
    auto idx = NewValueNode(SizeToLong(output_idx));
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(output_idx);
    idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
    auto getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
    auto abs = cnode->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abs);
    auto abs_i = abs->elements()[output_idx];
    MS_EXCEPTION_IF_NULL(abs_i);
    getitem->set_abstract(abs_i);
    const auto dev_fmt = AnfAlgo::GetOutputFormat(cnode, output_idx);
    const auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_idx);
    if (origin_type != device_type) {
      replace_node = AddCastOpNodeToGraph(func_graph, getitem, orig_cnode, dev_fmt, device_type, origin_type,
                                          origin_shape, origin_type, AnfAlgo::GetOutputReshapeType(getitem, 0));
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
      if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, output_idx)) {
        kernel_graph->ReplaceInternalOutput(cnode, replace_node, output_idx, 0);
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
}

AnfNodePtr InsertCastForOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_cnode, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) == 0) {
    return cnode;
  }
  MS_EXCEPTION_IF_NULL(cnode->Type());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  // Single output, output is TUPLE
  if (AnfUtils::IsRealKernel(cnode) &&
      AnfAlgo::GetOutputKernelObjectType(cnode, 0) == kernel::KernelObjectType::TUPLE) {
    // output is real tuple
    MS_LOG(INFO) << "The output's ObjectType is TUPLE, can not insert cast yet, skip it. Node: "
                 << cnode->fullname_with_scope();
    return cnode;
  }
  // Single output, output is not TUPLE
  if (!cnode->Type()->isa<Tuple>()) {
    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, 0);
    const abstract::BaseShapePtr origin_shape = AnfAlgo::GetOutputDetailShape(cnode, 0);
    const TypeId origin_type = common::AnfAlgo::GetOutputInferDataType(cnode, 0);
    const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, 0);
    AnfNodePtr replace_node = cnode;
    if (origin_type != device_type) {
      replace_node = AddCastOpNodeToGraph(func_graph, cnode, orig_cnode, dev_fmt, device_type, origin_type,
                                          origin_shape, origin_type, AnfAlgo::GetOutputReshapeType(cnode, 0));
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
      if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, 0)) {
        kernel_graph->ReplaceInternalOutput(cnode, replace_node);
      }
    }
    return replace_node;
  }
  // Multiple output
  return InsertCastForMultipleOutput(func_graph, orig_cnode, cnode);
}
}  // namespace

const BaseRef InsertCast::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertCast::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = InsertCastForInput(func_graph, cnode);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(node)) {
    kernel_graph->ReplaceInternalOutput(node, new_node);
  }
  // process output
  return InsertCastForOutput(func_graph, cnode, new_node);
}
}  // namespace opt
}  // namespace mindspore
