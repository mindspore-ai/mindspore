/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/tensor_array.h"

#include <memory>
#include <utility>
#include <vector>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
std::vector<std::string> TensorArrayAddFlowCond1::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimTensorArrayWrite->name());
  return ret;
}

const BaseRef TensorArrayAddFlowCond1::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VarPtr x4 = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorArrayWrite, x1, x2, x3, x4});
}

std::vector<std::string> TensorArrayAddFlowCond2::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimTensorArrayGather->name());
  return ret;
}

const BaseRef TensorArrayAddFlowCond2::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorArrayGather, x1, x2, x3});
}

const AnfNodePtr TensorArrayAddFlow::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto ta_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(ta_node);
  float flow_value = 0.0;
  // generate flow input
  auto flow_node = NewValueNode(MakeValue(flow_value));
  MS_EXCEPTION_IF_NULL(flow_node);
  // set abstract
  auto node_abstract = std::make_shared<abstract::AbstractScalar>(flow_value);
  flow_node->set_abstract(node_abstract);
  // add cnode input
  auto ta_node_inputs = ta_node->inputs();
  if (HasAbstractMonad(ta_node_inputs.back())) {
    auto input_size = ta_node_inputs.size();
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.assign(ta_node_inputs.begin(), ta_node_inputs.end());
    (void)new_inputs.insert(new_inputs.cbegin() + SizeToInt(input_size) - 1, flow_node);
    ta_node->set_inputs(new_inputs);
  } else {
    ta_node->add_input(flow_node);
  }
  return node;
}

std::vector<std::string> GeTensorArrayCastIndex::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimTensorArrayWrite->name());
  return ret;
}

const BaseRef GeTensorArrayCastIndex::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VarPtr x4 = std::make_shared<Var>();
  VarPtr x5 = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorArrayWrite, x1, x2, x3, x4, x5});
}

const AnfNodePtr GeTensorArrayCastIndex::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  const size_t index_input_index = 2;
  auto index_input_node = node->cast<CNodePtr>()->input(index_input_index);
  MS_EXCEPTION_IF_NULL(index_input_node);
  // Get cast prim
  auto cast_primitive = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_EXCEPTION_IF_NULL(cast_primitive);

  TypePtr src_type = TypeIdToType(TypeId::kNumberTypeInt64);
  TypePtr dst_type = TypeIdToType(TypeId::kNumberTypeInt32);
  MS_EXCEPTION_IF_NULL(src_type);
  MS_EXCEPTION_IF_NULL(dst_type);
  auto src_attr_value = MakeValue(src_type);
  auto dst_attr_value = MakeValue(dst_type);
  auto prim = std::make_shared<Primitive>(cast_primitive->AddAttr("dst_type", dst_attr_value));
  MS_EXCEPTION_IF_NULL(prim);
  prim = std::make_shared<Primitive>(prim->AddAttr("DstT", dst_attr_value));
  prim = std::make_shared<Primitive>(prim->AddAttr("SrcT", src_attr_value));

  // Insert cast
  auto type_node = NewValueNode(dst_type);
  MS_EXCEPTION_IF_NULL(type_node);
  type_node->set_abstract(dst_type->ToAbstract());

  auto new_node = graph->NewCNode({NewValueNode(prim), index_input_node, type_node});
  auto cast_abstract = index_input_node->abstract();
  MS_EXCEPTION_IF_NULL(cast_abstract);
  MS_EXCEPTION_IF_NULL(new_node);
  cast_abstract->set_type(dst_type);
  new_node->set_abstract(cast_abstract);

  auto input_names = std::vector<std::string>{"x", "dst_type"};
  auto output_names = std::vector<std::string>{"output"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), new_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), new_node);

  auto mgr = graph->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  mgr->SetEdge(node, index_input_index, new_node);
  return node;
}
}  // namespace opt
}  // namespace mindspore
