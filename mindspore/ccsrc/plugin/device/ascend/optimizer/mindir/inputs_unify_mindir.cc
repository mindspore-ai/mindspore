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

#include "plugin/device/ascend/optimizer/mindir/inputs_unify_mindir.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
namespace opt {

const AnfNodePtr InputsUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  if (GetCNodePrimitive(node) == nullptr) {
    return nullptr;
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto adpt = transform::FindAdapter(node);
  if (adpt == nullptr) {
    return nullptr;
  }

  auto input_map = adpt->getInputMap();
  for (auto it : input_map) {
    auto input = cnode->input(it.first);
    MS_EXCEPTION_IF_NULL(input);
    auto abstract = input->abstract();

    if (abstract->isa<abstract::AbstractScalar>()) {
      if (input->isa<ValueNode>()) {
        auto tensor_node = CreateScalarValueTensor(func_graph, input);
        manager->SetEdge(cnode, it.first, tensor_node);
      } else {
        // Insert ScalarToTensor
        auto tensor_node = CreateScalarToTensor(func_graph, input);
        manager->SetEdge(cnode, it.first, tensor_node);
      }
    } else if (abstract->isa<abstract::AbstractTuple>()) {
      // Insert TupleToTensor
      auto tensor_node = CreateTupleToTensor(func_graph, input);
      manager->SetEdge(cnode, it.first, tensor_node);
    }
  }
  return node;
}

abstract::AbstractBasePtr InputsUnifyMindIR::GenerateAbsByOpInfer(const CNodePtr &tuple_to_tensor) const {
  auto primitive = GetCNodePrimitive(tuple_to_tensor);
  MS_EXCEPTION_IF_NULL(primitive);
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (!found.has_value()) {
    MS_LOG(INTERNAL_EXCEPTION) << primitive->name() << "infer is not registered.";
  }

  auto input_list = tuple_to_tensor->inputs();
  std::vector<AbstractBasePtr> input_args;
  std::for_each(input_list.begin() + kSizeOne, input_list.end(),
                [&input_args](const auto &input) { input_args.emplace_back(input->abstract()); });
  auto infer_impl = found.value();
  auto abs = infer_impl.InferShapeAndType(nullptr, primitive, input_args);
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for " << primitive->name() << " is " << abs->ToString();
  return abs;
}

CNodePtr InputsUnifyMindIR::CreateTupleToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto prim = NewValueNode(std::make_shared<Primitive>(kTupleToTensorOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, node};
  CNodePtr tuple_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tuple_to_tensor);
  // attr dtype
  auto data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), tuple_to_tensor);

  // set abstract
  auto abs = GenerateAbsByOpInfer(tuple_to_tensor);
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TupleToTensor op is " << abs->ToString();
  tuple_to_tensor->set_abstract(abs);

  return tuple_to_tensor;
}

ValueNodePtr InputsUnifyMindIR::CreateScalarValueTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto value_ptr = GetValueNode(node);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto tensor = ScalarToTensor(value_ptr->cast<ScalarPtr>());
  auto const_value_node = NewValueNode(tensor);
  const_value_node->set_abstract(tensor->ToAbstract());
  func_graph->AddValueNode(const_value_node);
  return const_value_node;
}

CNodePtr InputsUnifyMindIR::CreateScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto prim = NewValueNode(std::make_shared<Primitive>(kScalarToTensorOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, node};
  CNodePtr scalar_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(scalar_to_tensor);

  // attr dtype
  auto data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), scalar_to_tensor);

  // set abstract
  auto abs = abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector{1}), TypeIdToType(data_type));
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for ScalarToTensor op is " << abs->ToString();
  scalar_to_tensor->set_abstract(abs);

  return scalar_to_tensor;
}
}  // namespace opt
}  // namespace mindspore
