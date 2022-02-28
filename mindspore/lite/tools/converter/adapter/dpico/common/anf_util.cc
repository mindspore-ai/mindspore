/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/anf_util.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include "common/op_enum.h"
#include "common/op_attr.h"
#include "common/string_util.h"
#include "ops/custom.h"
#include "ops/transpose.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr size_t kTupleGetItemInputSize = 3;
constexpr size_t kInputNodeOutputIndexInTupleGetItem = 2;
using PrimitiveCPtr = std::shared_ptr<ops::PrimitiveC>;
}  // namespace
bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(0), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}

STATUS GetPrimitiveType(const AnfNodePtr &node, std::string *name) {
  if (name == nullptr) {
    MS_LOG(ERROR) << "name is nulltr.";
    return RET_ERROR;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (CheckPrimitiveType(node, prim::kPrimCustom)) {
      auto custom_prim = utils::cast<std::shared_ptr<ops::Custom>>(primitive);
      MS_ASSERT(custom_prim != nullptr);
      *name = custom_prim->get_type();
      return RET_OK;
    } else {
      *name = primitive->name();
      return RET_OK;
    }
  } else if (node->isa<ValueNode>()) {
    auto fn_value = GetValueNode<PrimitivePtr>(node);
    *name = fn_value->name();
    return RET_OK;
  }
  MS_LOG(ERROR) << "There is no name for this node";
  return RET_ERROR;
}
STATUS GetShapeVectorFromParameter(const AnfNodePtr &anode, ShapeVector *shape_vector) {
  if (shape_vector == nullptr) {
    MS_LOG(ERROR) << "shape vector is nullptr.";
    return RET_ERROR;
  }
  if (!utils::isa<ParameterPtr>(anode)) {
    MS_LOG(ERROR) << "anode should be parameter node. ";
    return RET_ERROR;
  }
  auto param_node = anode->cast<ParameterPtr>();
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return lite::RET_PARAM_INVALID;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << param_node->name();
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << param_node->name();
    return lite::RET_PARAM_INVALID;
  }
  *shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}
std::vector<int> CastToInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<int> cur_value = {};
  if (utils::isa<ValueSequeuePtr>(value)) {
    if (!value->cast<ValueSequeuePtr>()->value().empty()) {
      if (value->cast<ValueSequeuePtr>()->value().front()->type()->number_type() == kNumberTypeInt64) {
        auto origin_value = GetValue<std::vector<int64_t>>(value);
        for (size_t index = 0; index < origin_value.size(); ++index) {
          cur_value.push_back(static_cast<int>(origin_value[index]));
        }
      } else {
        cur_value = GetValue<std::vector<int>>(value);
      }
    }
  } else {
    if (value->type()->number_type() == kNumberTypeInt64) {
      cur_value.push_back(static_cast<int>(GetValue<int64_t>(value)));
    } else {
      cur_value.push_back(GetValue<int>(value));
    }
  }
  return cur_value;
}
size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_ASSERT(tuple_get_item != nullptr);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(ERROR) << "The node tuple_get_item must have 2 inputs!";
    return -1;
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_ASSERT(output_index_value_node != nullptr);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  auto value_vec = CastToInt(value_node->value());
  if (value_vec.empty()) {
    MS_LOG(ERROR) << "value vec is empty.";
    return -1;
  }
  return IntToSize(value_vec.front());
}
STATUS GetOutputShapesFromCNode(const CNodePtr &cnode, std::vector<ShapeVector> *output_shapes) {
  AbstractBasePtr abstract = nullptr;
  if (CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    auto tuple_inputs = cnode->inputs();
    MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
    auto get_item_input_cnode = tuple_inputs.at(1);
    MS_ASSERT(get_item_input_cnode != nullptr);
    auto idx = GetTupleGetItemOutIndex(cnode);
    if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
      return RET_ERROR;
    }
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
    auto abstract_list = abstract_tuple->elements();
    if (abstract_list.size() <= idx) {
      MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
      return RET_ERROR;
    }
    abstract = abstract_list[idx];
  } else {
    abstract = cnode->abstract();
  }
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "abstract cnode is nullptr. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (utils::isa<abstract::AbstractTuplePtr>(abstract)) {
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract);
    auto abstract_list = abstract_tuple->elements();
    for (const auto &elem : abstract_list) {
      ShapeVector shape_vector;
      if (FetchShapeFromAbstract(elem, &shape_vector) != RET_OK) {
        MS_LOG(ERROR) << "fetch shape from abstract tuple elem failed. " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
      if (shape_vector.empty()) {
        MS_LOG(ERROR) << "shape_vector is empty." << cnode->fullname_with_scope();
        return RET_ERROR;
      }
      output_shapes->emplace_back(shape_vector);
    }
    return RET_OK;
  } else {
    ShapeVector shape_vector;
    if (FetchShapeFromAbstract(abstract, &shape_vector) != RET_OK) {
      MS_LOG(ERROR) << "fetch shape from abstract failed. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (shape_vector.empty()) {
      MS_LOG(ERROR) << "shape_vector is empty." << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    output_shapes->emplace_back(shape_vector);
  }
  return RET_OK;
}

STATUS GetInputShapeFromCNode(const mindspore::CNodePtr &cnode, size_t input_idx, ShapeVector *shape) {
  if (shape == nullptr) {
    MS_LOG(ERROR) << "shape is nullptr.";
    return RET_ERROR;
  }
  auto input_abstract = GetCNodeInputAbstract(cnode, input_idx);
  if (input_abstract == nullptr) {
    MS_LOG(ERROR) << "input_abstract is nullptr.";
    return RET_ERROR;
  }
  if (FetchShapeFromAbstract(input_abstract, shape) != RET_OK) {
    MS_LOG(ERROR) << "fetch shape from abstract failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape) {
  if (shape == nullptr) {
    MS_LOG(ERROR) << "shape is nullptr.";
    return RET_ERROR;
  }
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "shape of cnode's output is invalid.";
    return RET_ERROR;
  }
  *shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}
STATUS FetchTypeIdFromAbstract(const abstract::AbstractBasePtr &abstract, TypeId *type_id) {
  if (type_id == nullptr) {
    MS_LOG(ERROR) << "type id is nullptr.";
    return RET_ERROR;
  }
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor->element() == nullptr) {
    MS_LOG(ERROR) << "element of abstract_tensor is nullptr.";
    return RET_ERROR;
  }
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "type_ptr of abstract_tensor is nullptr.";
    return RET_ERROR;
  }
  *type_id = type_ptr->type_id();
  return RET_OK;
}

int GetAnfNodeOutputShape(const AnfNodePtr &input, ShapeVector *shape_vector) {
  if (shape_vector == nullptr) {
    MS_LOG(ERROR) << "shape vector is nullptr." << input->fullname_with_scope();
    return RET_ERROR;
  }
  if (utils::isa<ParameterPtr>(input)) {
    if (GetShapeVectorFromParameter(input, shape_vector) != RET_OK) {
      MS_LOG(ERROR) << "get output shape for preprocessor failed. " << input->fullname_with_scope();
      return RET_ERROR;
    }
  } else if (utils::isa<CNodePtr>(input)) {
    auto input_cnode = input->cast<CNodePtr>();
    std::vector<ShapeVector> output_shapes;
    if (GetOutputShapesFromCNode(input_cnode, &output_shapes) != RET_OK) {
      MS_LOG(ERROR) << "get output shapes from cnode failed. " << input_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (output_shapes.size() == 1) {
      *shape_vector = output_shapes.at(0);
    } else {
      MS_LOG(ERROR) << input_cnode->fullname_with_scope() << " has " << output_shapes.size()
                    << " output, which should be 1.";
      return RET_ERROR;
    }
  }
  if (shape_vector->empty()) {
    MS_LOG(ERROR) << "subgraph input shape shouldn't be empty. " << input->fullname_with_scope();
    return RET_ERROR;
  } else if (shape_vector->at(0) < 0) {
    MS_LOG(WARNING) << " the N axis of " << input->fullname_with_scope() << "'s output shape is " << shape_vector->at(0)
                    << ", which will be set to 1.";
    shape_vector->at(0) = 1;
  }
  return RET_OK;
}

std::string TypeIdToString(TypeId type_id) {
  const std::unordered_map<int, std::string> kTypeIdMap{
    {kNumberTypeFloat16, "Float16"}, {kNumberTypeFloat, "Float32"},    {kNumberTypeFloat32, "Float32"},
    {kNumberTypeInt8, "Int8"},       {kNumberTypeInt16, "Int16"},      {kNumberTypeInt, "Int32"},
    {kNumberTypeInt32, "Int32"},     {kNumberTypeUInt8, "UInt8"},      {kNumberTypeUInt16, "UInt16"},
    {kNumberTypeUInt, "UInt32"},     {kNumberTypeUInt32, "UInt32"},    {kObjectTypeString, "String"},
    {kNumberTypeBool, "Bool"},       {kObjectTypeTensorType, "Tensor"}};
  std::string type_str = "Unknown";
  if (kTypeIdMap.find(type_id) != kTypeIdMap.end()) {
    type_str = kTypeIdMap.at(type_id);
  }
  return type_str;
}

bool CheckInputs(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return false;
  }
  if (std::any_of(cnode->inputs().begin(), cnode->inputs().end(),
                  [](const AnfNodePtr &anf_node) { return anf_node == nullptr; })) {
    MS_LOG(ERROR) << "input is nullptr.";
    return false;
  }
  return true;
}
std::string GetCustomOutputName(const AnfNodePtr &node) {
  std::string output_name;
  auto input_cnode = node->cast<CNodePtr>();
  if (input_cnode == nullptr) {
    MS_LOG(ERROR) << "custom node should be cnode. " << node->fullname_with_scope();
    return "";
  }
  if (input_cnode->GetAttr(kOutputsNames) != nullptr) {
    auto output_names = GetValue<std::vector<std::string>>(input_cnode->GetAttr(kOutputsNames));
    if (output_names.size() == 1) {
      output_name = output_names.at(0);
    } else {
      MS_LOG(ERROR) << "multi-output's custom node shouldn't be a subgraph's input cnode. "
                    << node->fullname_with_scope();
      return "";
    }
  }
  return output_name;
}
tensor::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                   TypeId data_type) {
  tensor::TensorPtr tensor_info = nullptr;
  if (shape.empty() && data_size == abstract::TypeIdSize(data_type)) {
    ShapeVector scalar_shape = {1};
    tensor_info = std::make_shared<tensor::Tensor>(data_type, scalar_shape);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "new tensor init failed";
      return nullptr;
    }
    tensor_info->set_shape({});
  } else {
    tensor_info = std::make_shared<tensor::Tensor>(data_type, shape);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "new tensor init failed";
      return nullptr;
    }
  }
  if (data_size == 0) {
    return tensor_info;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "input tensor data is nullptr";
    return nullptr;
  }
  auto ret = memcpy_s(tensor_info->data_c(), tensor_info->data().nbytes(), data, data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error : " << ret;
    return nullptr;
  }
  return tensor_info;
}

AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, TypeId data_type) {
  auto tensor_info = dpico::CreateTensorInfo(nullptr, 0, shape, data_type);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto abstract = tensor_info->ToAbstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return nullptr;
  }
  return abstract;
}

int InitParameterFromTensorInfo(const ParameterPtr &param_node, const tensor::TensorPtr &tensor_info) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor info is nullptr.";
    return RET_ERROR;
  }
  auto abstract_tensor = tensor_info->ToAbstract();
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Create abstract tensor failed.";
    return RET_ERROR;
  }
  param_node->set_abstract(abstract_tensor);
  param_node->set_default_param(tensor_info);
  return RET_OK;
}

abstract::AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (index >= inputs.size()) {
    MS_LOG(ERROR) << "index: " << index << " is greater than inputs size " << inputs.size();
    return nullptr;
  }
  auto input = inputs[index];
  if (input == nullptr) {
    MS_LOG(ERROR) << "CNode input is nullptr";
    return nullptr;
  }

  abstract::AbstractBasePtr abstract = nullptr;
  if (utils::isa<ParameterPtr>(input)) {
    auto parameter = input->cast<ParameterPtr>();
    abstract = parameter->abstract();
  } else if (utils::isa<CNodePtr>(input)) {
    auto input_cnode = input->cast<CNodePtr>();
    if (CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
      auto tuple_inputs = input_cnode->inputs();
      MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
      auto get_item_input_cnode = tuple_inputs.at(1);
      MS_ASSERT(get_item_input_cnode != nullptr);
      auto idx = GetTupleGetItemOutIndex(input_cnode);
      if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
        MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
        return nullptr;
      }
      auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
      auto abstract_list = abstract_tuple->elements();
      if (abstract_list.size() <= idx) {
        MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
        return nullptr;
      }
      abstract = abstract_list[idx];
    } else {
      abstract = input_cnode->abstract();
    }
  } else {
    MS_LOG(ERROR) << "unsupported input node type";
    return nullptr;
  }
  return abstract;
}

abstract::AbstractBasePtr GetAbstractFromAnfNode(const AnfNodePtr &node) {
  AbstractBasePtr abstract = nullptr;
  if (utils::isa<ParameterPtr>(node)) {
    auto parameter = node->cast<ParameterPtr>();
    abstract = parameter->abstract();
  } else if (utils::isa<CNodePtr>(node)) {
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
      auto tuple_inputs = cnode->inputs();
      MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
      auto get_item_input_cnode = tuple_inputs.at(1);
      MS_ASSERT(get_item_input_cnode != nullptr);
      auto idx = GetTupleGetItemOutIndex(cnode);
      if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
        MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
        return nullptr;
      }
      auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
      auto abstract_list = abstract_tuple->elements();
      if (abstract_list.size() <= idx) {
        MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
        return nullptr;
      }
      abstract = abstract_list[idx];
    } else {
      abstract = cnode->abstract();
    }
  }
  return abstract;
}

ParameterPtr BuildIntValueParameterNode(const api::FuncGraphPtr &func_graph, const int32_t &data,
                                        const std::string &node_name) {
  MS_ASSERT(func_graph != nullptr);
  auto param_node = func_graph->add_parameter();
  param_node->set_name(node_name);

  auto tensor_info = CreateTensorInfo(&data, sizeof(int32_t), {1}, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildIntVecParameterNode(const api::FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                      const std::string &node_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(data.size() != 0);
  auto param_node = func_graph->add_parameter();
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto tensor_info = CreateTensorInfo(data.data(), data.size() * sizeof(int32_t), shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return param_node;
}

ParameterPtr BuildIntVec2DParameterNode(const api::FuncGraphPtr &func_graph,
                                        const std::vector<std::vector<int32_t>> &data, const std::string &node_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(data.size() != 0);
  auto param_node = func_graph->add_parameter();
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector;
  shape_vector.push_back(data.size());
  shape_vector.push_back(kDims2);

  std::vector<int32_t> data_1d;
  for (auto pair : data) {
    data_1d.insert(data_1d.end(), pair.begin(), pair.end());
  }

  auto size = data_1d.size() * sizeof(int32_t);
  auto tensor_info = CreateTensorInfo(data_1d.data(), size, shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildFloatValueParameterNode(const api::FuncGraphPtr &func_graph, const float &data,
                                          const std::string &node_name) {
  MS_ASSERT(func_graph != nullptr);
  auto param_node = func_graph->add_parameter();
  param_node->set_name(node_name);

  auto tensor_info = CreateTensorInfo(&data, sizeof(float), {1}, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

CNodePtr GenTransposeNode(const api::FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                          const std::vector<int> &perm, const std::string &cnode_name) {
  MS_ASSERT(func_graph != nullptr && input_node != nullptr);
  auto perm_node = BuildIntVecParameterNode(func_graph, perm, cnode_name + "_perm");
  if (perm_node == nullptr) {
    MS_LOG(ERROR) << "new perm_node error";
    return nullptr;
  }
  auto trans_prim = std::make_shared<ops::Transpose>();
  if (trans_prim == nullptr) {
    MS_LOG(ERROR) << "new trans_prim failed";
    return nullptr;
  }
  auto cnode = func_graph->NewCNode(trans_prim, {input_node, perm_node});
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return nullptr;
  }
  auto manager = api::FuncGraphManager::Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return nullptr;
  }
  manager->SetEdge(cnode, 1, input_node);
  manager->SetEdge(cnode, kInputIndex2, perm_node);
  cnode->set_fullname_with_scope(cnode_name);
  return cnode;
}

tensor::TensorPtr GetTensorInfo(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  if (!utils::isa<ParameterPtr>(node)) {
    if (utils::isa<ValueNodePtr>(node)) {
      auto valueNode = node->cast<ValueNodePtr>();
      auto value = std::dynamic_pointer_cast<tensor::Tensor>(valueNode->value());
      if (value != nullptr) {
        return value;
      }
    }
    MS_LOG(DEBUG) << "get lite param value node neither parameter node or value node";
    return nullptr;
  }
  auto param = node->cast<ParameterPtr>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(param->default_param());
  return tensor_info;
}

std::vector<std::vector<int>> CastToVec2DInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }

  std::vector<std::vector<int>> result_value;
  if (utils::isa<ValueSequeuePtr>(value)) {
    if (value->cast<ValueSequeuePtr>()
          ->value()
          .front()
          ->cast<ValueSequeuePtr>()
          ->value()
          .front()
          ->type()
          ->number_type() == kNumberTypeInt64) {
      auto origin_value = GetValue<std::vector<std::vector<int64_t>>>(value);
      for (size_t i = 0; i < origin_value.size(); ++i) {
        std::vector<int> cur_value;
        for (size_t j = 0; j < origin_value.at(i).size(); ++j) {
          cur_value.push_back(static_cast<int>(origin_value[i][j]));
        }
        result_value.push_back(cur_value);
      }
    } else {
      result_value = GetValue<std::vector<std::vector<int>>>(value);
    }
  }
  return result_value;
}

bool GetBoolAttr(const AnfNodePtr &node, const std::string &attr_name) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cur node is not a cnode. " << node->fullname_with_scope();
    return false;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr:" << cnode->fullname_with_scope();
    return false;
  }
  auto value_ptr = primitive->GetAttr(attr_name);
  if (value_ptr == nullptr) {
    MS_LOG(ERROR) << "There is no attr named " << attr_name << " for node " << cnode->fullname_with_scope();
    return false;
  }
  return GetValue<bool>(value_ptr);
}

STATUS GetDataTypeAndShape(const ParameterPtr &param_node, TypeId *data_type, ShapeVector *shape_vector) {
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "param node is nullptr.";
    return RET_ERROR;
  }
  if (data_type == nullptr) {
    MS_LOG(ERROR) << "data type is nullptr.";
    return RET_ERROR;
  }
  if (shape_vector == nullptr) {
    MS_LOG(ERROR) << "shape vector is nullptr.";
    return RET_ERROR;
  }
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << param_node->name();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
  auto typePtr = abstract_tensor->element()->GetTypeTrack();
  MS_ASSERT(typePtr != nullptr);
  *data_type = typePtr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << param_node->name();
    return RET_ERROR;
  }
  *shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}

STATUS GetShapeVectorFromStringTensor(const tensor::TensorPtr &tensor_info, ShapeVector *shape_vector, size_t *offset) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor info is nullptr.";
    return RET_ERROR;
  }
  if (shape_vector == nullptr) {
    MS_LOG(ERROR) << "shape vector is nullptr.";
    return RET_ERROR;
  }
  if (offset == nullptr) {
    MS_LOG(ERROR) << "offset is nullptr.";
    return RET_ERROR;
  }
  auto data_type = tensor_info->data_type();
  if (data_type != kObjectTypeString) {
    MS_LOG(ERROR) << "This function only used for string tensor.";
    return RET_ERROR;
  }
  shape_vector->clear();
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  std::string shape_str;
  std::string shape_size_str;
  *offset = 0;
  size_t cnt = 0;
  for (; *offset < tensor_info->Size(); (*offset)++) {
    if (tensor_data[*offset] == ',') {
      (*offset)++;
      break;
    }
    shape_size_str.push_back(tensor_data[*offset]);
  }
  if (*offset == 0) {
    MS_LOG(ERROR) << "string tensor's dim size not found.";
    return RET_ERROR;
  }
  if (!IsValidUnsignedNum(shape_size_str)) {
    MS_LOG(ERROR) << "shape_size str must an unsigned int.";
    return RET_ERROR;
  }
  size_t shape_size = std::stoi(shape_size_str);
  for (; *offset < tensor_info->Size(); (*offset)++) {
    if (tensor_data[*offset] == ',') {
      cnt++;
      if (!IsValidUnsignedNum(shape_str)) {
        MS_LOG(ERROR) << "shape str must an unsigned int.";
        return RET_ERROR;
      }
      shape_vector->push_back(std::stoi(shape_str));
      shape_str.clear();
    } else {
      shape_str.push_back(tensor_data[*offset]);
    }
    if (cnt == shape_size) {
      (*offset)++;
      break;
    }
  }
  if (shape_vector->empty()) {
    MS_LOG(ERROR) << "string tensor's shape shouldn't be empty.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
