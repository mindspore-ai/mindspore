/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/maketuple_unify_mindir.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"

/* This pass changes the following pattern.

  MakeTuple and Reshape operations using Scalars.
    ###############
    Pattern:
    Scalar -> MakeTuple -> Tuple(Scalar) -> Reshape -> Tensor

    Replace:
    ScalarToTensor -> MakeTuple -> Tuple(Tensor) -> Concat -> Reshape -> Tensor
    ###############

    MakeTuple's Scalar inputs will be converted to Tensors, followed by a Concat op to allow the reshape by a Tensor.
    ScalarToTensor conversion op is inserted before to make sure the old/new pattern inputs/outputs' types are agree.
*/

namespace mindspore {
namespace opt {

const BaseRef MakeTupleUnifyMindIR::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef make_tuple({std::make_shared<Primitive>(kMakeTupleOpName), Xs});
  return VectorRef({std::make_shared<Primitive>(kReshapeOpName), x, make_tuple});
}

const AnfNodePtr MakeTupleUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto reshape_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  auto make_tuple_node = reshape_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  auto make_tuple_cnode = make_tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  auto abstract = make_tuple_cnode->input(kIndex1)->abstract();
  if (!utils::isa<abstract::AbstractScalarPtr>(abstract)) {
    return node;
  }
  auto type_ptr = abstract->cast<abstract::AbstractScalarPtr>()->GetTypeTrack();
  TypeId type_id = type_ptr->type_id();

  std::vector<AnfNodePtr> tensor_node_list = {NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
  abstract::BaseShapePtrList tuple_shape_list;
  TypePtrList tuple_type_list;
  ShapeVector scalar_shape = {kSizeOne};
  for (size_t i = kIndex1; i < make_tuple_cnode->size(); i++) {
    auto make_tuple_input = make_tuple_cnode->input(i);

    auto abstract_scalar = make_tuple_input->abstract()->cast<abstract::AbstractScalarPtr>();
    if (abstract_scalar == nullptr) {
      return node;
    }
    auto item_type_ptr = abstract_scalar->GetTypeTrack();
    TypeId item_type_id = item_type_ptr->type_id();
    if (item_type_id != type_id) {
      return node;
    }
    tuple_type_list.push_back(TypeIdToType(type_id));

    // Insert ScalarToTensor before MakeTuple
    if (make_tuple_input->isa<ValueNode>()) {
      auto tensor_node = CreateScalarValueTensor(func_graph, make_tuple_input);
      tensor_node_list.push_back(tensor_node);
    } else {
      auto tensor_node = CreateScalarToTensor(func_graph, make_tuple_input, type_id);
      tensor_node_list.push_back(tensor_node);
    }
    tuple_shape_list.push_back(std::make_shared<abstract::Shape>(scalar_shape));
  }
  auto new_make_tuple_cnode = func_graph->NewCNode(tensor_node_list);

  // Apply modified abstract to MakeTuple
  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(tuple_shape_list),
                                             std::make_shared<Tuple>(tuple_type_list));
  new_make_tuple_cnode->set_abstract(tmp_abstract);

  // Insert concat after MakeTuple
  auto concat_node = CreateConcat(func_graph, new_make_tuple_cnode,
                                  make_tuple_cnode->cast<CNodePtr>()->fullname_with_scope() + "_concat_make_tuple");
  auto primitive = GetCNodePrimitive(concat_node);
  MS_EXCEPTION_IF_NULL(primitive);
  int64_t num_of_inputs = static_cast<int64_t>(new_make_tuple_cnode->inputs().size() - kSizeOne);
  std::vector<int64_t> dyn_input_size_empty{num_of_inputs};
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(num_of_inputs), concat_node);
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size_empty), concat_node);

  auto concat_abstract =
    abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector({num_of_inputs})), TypeIdToType(type_id));
  concat_node->set_abstract(concat_abstract);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(reshape_cnode, kIndex2, concat_node);
  return reshape_cnode;
}

ValueNodePtr MakeTupleUnifyMindIR::CreateScalarValueTensor(const FuncGraphPtr &func_graph,
                                                           const AnfNodePtr &node) const {
  auto value_ptr = GetValueNode(node);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto tensor = ScalarToTensor(value_ptr->cast<ScalarPtr>());
  auto const_value_node = NewValueNode(tensor);
  const_value_node->set_abstract(tensor->ToAbstract());
  func_graph->AddValueNode(const_value_node);
  return const_value_node;
}

CNodePtr MakeTupleUnifyMindIR::CreateScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    TypeId type_id) const {
  auto prim = NewValueNode(std::make_shared<Primitive>(kScalarToTensorOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, node};
  CNodePtr scalar_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(scalar_to_tensor);

  auto primitive = GetCNodePrimitive(scalar_to_tensor);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->set_attr("dtype", TypeIdToType(type_id));

  // set abstract
  ShapeVector tensor_shape = {1};
  auto tensor_shape_ptr = std::make_shared<abstract::Shape>(tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor_shape_ptr);

  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape), TypeIdToType(type_id));
  MS_EXCEPTION_IF_NULL(tmp_abstract);
  scalar_to_tensor->set_abstract(tmp_abstract);
  return scalar_to_tensor;
}

CNodePtr MakeTupleUnifyMindIR::CreateConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &maketuple_node,
                                            const std::string &cnode_name, int64_t axis) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(kConcatOpName)), maketuple_node};
  auto concat = NewCNode(concat_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(concat);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(axis), concat);
  concat->set_fullname_with_scope(cnode_name);
  return concat;
}
}  // namespace opt
}  // namespace mindspore
