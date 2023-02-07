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

#include "tools/converter/adapter/acl/common/utils.h"
#include <functional>
#include "tools/optimizer/common/gllo_utils.h"
#include "base/base_ref.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/utils.h"
#include "src/common/log_util.h"
#include "ir/func_graph.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace acl {
namespace {
constexpr size_t kTupleGetItemInputSize = 3;
constexpr size_t kSecondIndex = 1;
constexpr size_t kInvalidSize = SIZE_MAX;
}  // namespace

static size_t GetTupleGetItemOutIndex(const mindspore::CNodePtr &tuple_get_item) {
  MS_CHECK_TRUE_MSG(tuple_get_item != nullptr, kInvalidSize, "tuple_get_item is nullptr.");
  MS_CHECK_TRUE_MSG(tuple_get_item->size() == mindspore::kTupleGetItemInputSize, kInvalidSize,
                    "The node tuple_get_item must have 3 inputs!");
  auto output_index_value_node = tuple_get_item->input(mindspore::kInputNodeOutputIndexInTupleGetItem);
  MS_CHECK_TRUE_MSG(output_index_value_node != nullptr, kInvalidSize, "output_index_value_node is nullptr.");
  auto value_node = output_index_value_node->cast<mindspore::ValueNodePtr>();
  MS_CHECK_TRUE_MSG(value_node != nullptr, kInvalidSize, "value_node is nullptr.");
  auto values = opt::CastToInt(value_node->value());
  MS_CHECK_TRUE_MSG(values.size() > 0, kInvalidSize, "value_node has no value.");
  return IntToSize(values.front());
}

static bool CheckPrimitiveType(const mindspore::AnfNodePtr &node, const mindspore::PrimitivePtr &primitive_type) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  if (node->isa<mindspore::CNode>()) {
    auto cnode = node->cast<mindspore::CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr.");
    return IsPrimitive(cnode->input(0), primitive_type);
  } else if (node->isa<mindspore::ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}

STATUS GetShapeVectorFromCNode(const mindspore::CNodePtr &cnode, std::vector<int64_t> *shape_vector) {
  mindspore::AbstractBasePtr cnode_abstract;
  if (CheckPrimitiveType(cnode, mindspore::prim::kPrimTupleGetItem)) {
    auto tuple_inputs = cnode->inputs();
    MS_CHECK_TRUE_MSG(tuple_inputs.size() == kTupleGetItemInputSize, lite::RET_ERROR, "The node must have 3 inputs!");
    auto get_item_input_cnode = tuple_inputs.at(kSecondIndex);
    MS_CHECK_TRUE_MSG(get_item_input_cnode != nullptr, lite::RET_ERROR, "input node is nullptr.");
    auto idx = GetTupleGetItemOutIndex(cnode);
    if (!mindspore::utils::isa<mindspore::abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple, cnode name: "
                    << get_item_input_cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    auto abstract_tuple =
      mindspore::utils::cast<mindspore::abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
    auto abstract_list = abstract_tuple->elements();
    if (abstract_list.size() <= idx) {
      MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
      return lite::RET_ERROR;
    }
    cnode_abstract = abstract_list[idx];
  } else {
    cnode_abstract = cnode->abstract();
  }
  CHECK_NULL_RETURN(cnode_abstract);
  if (cnode_abstract->BuildShape() == mindspore::abstract::kNoShape) {
    *shape_vector = std::vector<int64_t>();
    return lite::RET_OK;
  }
  if (!mindspore::utils::isa<mindspore::abstract::AbstractTensorPtr>(cnode_abstract)) {
    MS_LOG(ERROR) << "Abstract is not abstract tensor. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto cnode_abstract_tensor = cnode_abstract->cast<mindspore::abstract::AbstractTensorPtr>();
  CHECK_NULL_RETURN(cnode_abstract_tensor);
  if (!mindspore::utils::isa<mindspore::abstract::ShapePtr>(cnode_abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of abstract tensor should be ShapePtr. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto shape_ptr = mindspore::utils::cast<mindspore::abstract::ShapePtr>(cnode_abstract_tensor->BuildShape());
  CHECK_NULL_RETURN(shape_ptr);
  if (shape_ptr->shape().empty()) {
    MS_LOG(INFO) << "Shape is empty " << cnode->fullname_with_scope();
  }

  *shape_vector = shape_ptr->shape();
  return lite::RET_OK;
}

TypeId GetTypeFromNode(const AnfNodePtr &node, const size_t tuple_idx) {
  TypeId type = kNumberTypeFloat32;
  MS_CHECK_TRUE_MSG(node != nullptr, type, "node is nullptr.");
  if (utils::isa<CNodePtr>(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, type, "cnode is nullptr.");
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = cnode->abstract()->cast<abstract::AbstractTensorPtr>();
      if (abstract_tensor == nullptr || abstract_tensor->element() == nullptr) {
        MS_LOG(WARNING) << "Abstract_tensor or abstract_tensor->element() is nullptr.";
        return type;
      }
      auto type_ptr = abstract_tensor->element()->GetTypeTrack();
      MS_CHECK_TRUE_MSG(type_ptr != nullptr, type, "type_ptr is nullptr.");
      type = type_ptr->type_id();
    } else if (utils::isa<abstract::AbstractTuplePtr>(cnode->abstract())) {
      auto abstract_tuple = cnode->abstract()->cast<abstract::AbstractTuplePtr>();
      if (abstract_tuple->elements().empty()) {
        MS_LOG(ERROR) << "abstract_tuple elements is empty.";
        return type;
      }
      if (tuple_idx >= abstract_tuple->size()) {
        MS_LOG(ERROR) << "tuple_idx out of range.";
        return type;
      }
      auto abstract_base = abstract_tuple->elements()[tuple_idx];
      if (utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
        auto abstract_tensor = abstract_base->cast<abstract::AbstractTensorPtr>();
        if (abstract_tensor == nullptr || abstract_tensor->element() == nullptr) {
          MS_LOG(WARNING) << "Abstract_tensor or abstract_tensor->element() is nullptr";
          return type;
        }
        auto type_ptr = abstract_tensor->element()->GetTypeTrack();
        MS_CHECK_TRUE_MSG(type_ptr != nullptr, type, "type_ptr is nullptr");
        type = type_ptr->type_id();
      }
    }
    MS_LOG(INFO) << "node type id is " << type;
  }
  return type;
}

std::vector<int> GetIntParameterData(const ParameterPtr &param_ptr) {
  std::vector<int> result;
  MS_CHECK_TRUE_MSG(param_ptr != nullptr, result, "Param is nullptr.");

  if (!param_ptr->has_default()) {
    MS_LOG(DEBUG) << "Param has not default.";
    return result;
  }
  auto default_param = param_ptr->default_param();
  MS_CHECK_TRUE_MSG(default_param != nullptr, result, "default_param is nullptr.");
  if (!utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "Tensor info is not tensor::TensorPtr.";
    return result;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  MS_CHECK_TRUE_MSG(default_param_ptr != nullptr, result, "default_param_ptr is nullptr.");
  if (default_param_ptr->data_type() != kNumberTypeInt32 && default_param_ptr->data_type() != kNumberTypeInt) {
    MS_LOG(DEBUG) << "Default param is not int.";
    return result;
  }

  auto ptr = reinterpret_cast<int *>(default_param_ptr->data_c());
  MS_CHECK_TRUE_MSG(ptr != nullptr, result, "ptr is nullptr.");
  int shape_size =
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<int>());
  for (int i = 0; i < shape_size; i++) {
    result.emplace_back(ptr[i]);
  }
  return result;
}

std::vector<float> GetFloatParameterData(const ParameterPtr &param_ptr) {
  std::vector<float> result;
  MS_CHECK_TRUE_MSG(param_ptr != nullptr, result, "Param is nullptr.");

  if (!param_ptr->has_default()) {
    MS_LOG(DEBUG) << "Param has not default.";
    return result;
  }
  auto default_param = param_ptr->default_param();
  MS_CHECK_TRUE_MSG(default_param != nullptr, result, "default_param is nullptr.");
  if (!utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "Tensor info is not tensor::TensorPtr.";
    return result;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  MS_CHECK_TRUE_MSG(default_param_ptr != nullptr, result, "default_param_ptr is nullptr.");
  if (default_param_ptr->data_type() != kNumberTypeFloat32 && default_param_ptr->data_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "Default param is not int.";
    return result;
  }

  auto ptr = reinterpret_cast<float *>(default_param_ptr->data_c());
  MS_CHECK_TRUE_MSG(ptr != nullptr, result, "ptr is nullptr.");
  int shape_size =
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<float>());
  for (int i = 0; i < shape_size; i++) {
    result.emplace_back(ptr[i]);
  }
  return result;
}

bool IsCaseNode(const CNodePtr node) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  if (node->input(0) == nullptr) {
    MS_LOG(WARNING) << "The input of node is nullptr.";
    return false;
  }
  if (!node->inputs().empty() && node->input(0)->isa<CNode>() &&
      GetCNodeFuncName(node->input(0)->cast<CNodePtr>()) == "switch_layer") {
    return true;
  }
  return false;
}

std::string GetCNodeTargetFuncName(const CNodePtr &cnode) {
  if (IsCaseNode(cnode)) {
    return string("Case");
  }
  auto name = GetCNodeFuncName(cnode);
  if (name == "switch_layer") {
    name = "";
  }
  return name;
}

STATUS DelRedundantParameter(const FuncGraphPtr &func_graph) {
  CHECK_NULL_RETURN(func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  auto parameters = func_graph->parameters();
  for (auto &parameter : parameters) {
    CHECK_NULL_RETURN(parameter);
    if (std::find(nodes.begin(), nodes.end(), parameter) == nodes.end()) {
      func_graph->DropNode(parameter);
    }
  }
  return lite::RET_OK;
}
}  // namespace acl
}  // namespace lite
}  // namespace mindspore
