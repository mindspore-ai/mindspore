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
#include "pre_activate/ascend/format_type/insert_cast.h"

#include <memory>
#include <string>
#include <vector>

#include "device/kernel_info.h"
#include "pre_activate/ascend/ascend_helper.h"
#include "pre_activate/common/helper.h"
#include "kernel/kernel_build_info.h"
#include "kernel/oplib/oplib.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr InsertCastForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs;
  AbstractBasePtrList abstract_list;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t output_idx = 0; output_idx < AnfAlgo::GetOutputTensorNum(cnode); ++output_idx) {
    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, output_idx);
    const std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(cnode, output_idx);
    const TypeId origin_type = AnfAlgo::GetOutputInferDataType(cnode, output_idx);
    const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_idx);
    auto idx = NewValueNode(SizeToInt(output_idx));
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int32Imm>(output_idx);
    idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
    auto getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
    AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, getitem.get());
    AnfNodePtr replace_node = nullptr;
    if (origin_type != device_type) {
      replace_node =
        AddCastOpNodeToGraph(func_graph, getitem, dev_fmt, device_type, origin_type, origin_shape, origin_type);
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
    } else {
      replace_node = getitem;
    }
    abstract_list.push_back(replace_node->abstract());
    make_tuple_inputs.push_back(replace_node);
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple;
}

AnfNodePtr InsertCastForOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) == 0) {
    return cnode;
  }
  MS_EXCEPTION_IF_NULL(cnode->Type());
  // Single output
  if (!cnode->Type()->isa<Tuple>()) {
    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, 0);
    std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(cnode, 0);
    const TypeId origin_type = AnfAlgo::GetOutputInferDataType(cnode, 0);
    const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, 0);
    AnfNodePtr replace_node = cnode;
    if (origin_type != device_type) {
      replace_node =
        AddCastOpNodeToGraph(func_graph, cnode, dev_fmt, device_type, origin_type, origin_shape, origin_type);
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
    }
    return replace_node;
  }
  // Multiple output
  return InsertCastForMultipleOutput(func_graph, cnode);
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
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = InsertCastForInput(func_graph, cnode);
  // process output
  return InsertCastForOutput(func_graph, new_node);
}
}  // namespace opt
}  // namespace mindspore
