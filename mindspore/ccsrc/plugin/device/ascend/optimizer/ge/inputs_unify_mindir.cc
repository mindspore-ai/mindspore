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

#include "plugin/device/ascend/optimizer/ge/inputs_unify_mindir.h"
#include <map>
#include <vector>
#include <memory>
#include "mindspore/core/ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
namespace opt {

const std::map<TypeId, TypeId> kReduceRaiseMap = {{kNumberTypeInt64, kNumberTypeInt32},
                                                  {kNumberTypeFloat64, kNumberTypeFloat32}};

template <typename T, typename D>
std::vector<D> CastVector(std::vector<T> src) {
  std::vector<D> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = static_cast<D>(src[i]);
  }
  return dst;
}

tensor::TensorPtr CastValueTensor(const tensor::TensorPtr &src, const TypePtr &dst_type) {
  TypeId src_type_id = src->data_type();
  TypeId dst_type_id = dst_type->type_id();
  if (src_type_id == kNumberTypeInt64 && dst_type_id == kNumberTypeInt32) {
    auto vec = TensorValueToVector<int64_t>(src);
    return std::make_shared<tensor::Tensor>(CastVector<int64_t, int32_t>(vec), dst_type);
  } else if (src_type_id == kNumberTypeFloat64 && dst_type_id == kNumberTypeFloat32) {
    auto vec = TensorValueToVector<double>(src);
    return std::make_shared<tensor::Tensor>(vec, dst_type);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not convert data type from " << TypeIdToString(src_type_id) << " to "
                               << TypeIdToString(dst_type_id);
  }
  return nullptr;
}

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

    AnfNodePtr tensor_node = input;
    if (input->isa<ValueNode>()) {
      tensor_node = CreateValueTensor(func_graph, input);
    } else if (abstract->isa<abstract::AbstractScalar>()) {
      tensor_node = CreateScalarToTensor(func_graph, input);
    } else if (abstract->isa<abstract::AbstractTuple>()) {
      tensor_node = CreateTupleToTensor(func_graph, input);
    }
    auto src_type = common::AnfAlgo::GetOutputInferDataType(tensor_node, 0);
    auto src_type_it = std::find(it.second.supported_dtypes.begin(), it.second.supported_dtypes.end(),
                                 transform::TransformUtil::ConvertDataType(src_type));
    if (src_type_it == it.second.supported_dtypes.end()) {
      auto iter = kReduceRaiseMap.find(src_type);
      if (iter == kReduceRaiseMap.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << cnode->fullname_with_scope() << " input(" << it.first
                                   << ") data type can not add Cast.";
      }
      auto dst_type_it = std::find(it.second.supported_dtypes.begin(), it.second.supported_dtypes.end(),
                                   transform::TransformUtil::ConvertDataType(iter->second));
      if (dst_type_it == it.second.supported_dtypes.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << cnode->fullname_with_scope() << " input(" << it.first
                                   << ") data type is not support.";
      }
      MS_LOG(INFO) << "Convert data type from " << TypeIdToString(src_type) << " to " << TypeIdToString(iter->second);
      tensor_node = CreateCastNode(func_graph, tensor_node, iter->second);
    }
    manager->SetEdge(cnode, it.first, tensor_node);
  }
  return node;
}

ValueNodePtr InputsUnifyMindIR::CreateValueTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto value = GetValueNode(node);
  MS_EXCEPTION_IF_NULL(value);
  tensor::TensorPtr tensor = nullptr;
  if (value->isa<Scalar>()) {
    tensor = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueSequence>()) {
    tensor = SequenceToTensor(value->cast<ValueSequencePtr>());
  } else if (value->isa<tensor::Tensor>()) {
    tensor = value->cast<tensor::TensorPtr>();
  }
  MS_EXCEPTION_IF_NULL(tensor);
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

CNodePtr InputsUnifyMindIR::CreateTupleToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto prim = std::make_shared<Primitive>(kTupleToTensorOpName);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {NewValueNode(prim), node};
  CNodePtr tuple_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tuple_to_tensor);
  // attr dtype
  auto data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), tuple_to_tensor);

  // set abstract
  auto abs = InferAbstract(prim, {node});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TupleToTensor op is " << abs->ToString();
  tuple_to_tensor->set_abstract(abs);

  return tuple_to_tensor;
}

AnfNodePtr InputsUnifyMindIR::CreateCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             TypeId data_type) const {
  if (node->isa<ValueNode>()) {
    auto value = GetValueNode(node);
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto new_tensor = CastValueTensor(tensor, TypeIdToType(data_type));
    MS_EXCEPTION_IF_NULL(new_tensor);
    auto const_value_node = NewValueNode(new_tensor);
    const_value_node->set_abstract(new_tensor->ToAbstract());
    return const_value_node;
  }

  auto prim = NewValueNode(std::make_shared<Primitive>(kCastOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, node, NewValueNode(TypeIdToType(data_type))};
  CNodePtr cast = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cast);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), cast);
  auto abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  abstract->set_type(TypeIdToType(data_type));
  cast->set_abstract(abstract);
  return cast;
}
}  // namespace opt
}  // namespace mindspore
