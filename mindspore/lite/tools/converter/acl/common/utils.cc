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

#include "tools/converter/acl/common/utils.h"
#include <functional>
#include "tools/optimizer/common/gllo_utils.h"
#include "base/base_ref.h"
#include "base/core_ops.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace acl {
namespace {
constexpr size_t kTupleGetItemInputSize = 3;
constexpr size_t kSecondIndex = 1;
constexpr size_t kInvalidSize = SIZE_MAX;
}  // namespace

static size_t GetTupleGetItemOutIndex(const mindspore::CNodePtr &tuple_get_item) {
  MS_ASSERT(tuple_get_item != nullptr);
  if (tuple_get_item->size() != mindspore::kTupleGetItemInputSize) {
    MS_LOG(ERROR) << "The node tuple_get_item must have 2 inputs!";
    return kInvalidSize;
  }
  auto output_index_value_node = tuple_get_item->input(mindspore::kInputNodeOutputIndexInTupleGetItem);
  MS_ASSERT(output_index_value_node != nullptr);
  auto value_node = output_index_value_node->cast<mindspore::ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  return IntToSize(opt::CastToInt(value_node->value()).front());
}

static bool CheckPrimitiveType(const mindspore::AnfNodePtr &node, const mindspore::PrimitivePtr &primitive_type) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<mindspore::CNode>()) {
    auto cnode = node->cast<mindspore::CNodePtr>();
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
    MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
    auto get_item_input_cnode = tuple_inputs.at(kSecondIndex);
    MS_ASSERT(get_item_input_cnode != nullptr);
    auto idx = GetTupleGetItemOutIndex(cnode);
    if (!mindspore::utils::isa<mindspore::abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
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
    MS_LOG(WARNING) << "Shape is empty " << cnode->fullname_with_scope();
  }

  *shape_vector = shape_ptr->shape();
  return lite::RET_OK;
}

TypeId GetTypeFromNode(const AnfNodePtr &node) {
  TypeId type = kNumberTypeFloat32;
  if (utils::isa<CNodePtr>(node)) {
    auto cnode = node->cast<CNodePtr>();
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      if (abstract_tensor == nullptr || abstract_tensor->element() == nullptr) {
        MS_LOG(WARNING) << "Abstract_tensor or abstract_tensor->element() is nullptr.";
        return type;
      }
      auto type_ptr = abstract_tensor->element()->GetTypeTrack();
      type = type_ptr->type_id();
    }
    MS_LOG(INFO) << "node type id is " << type;
  }
  return type;
}

std::vector<int> GetIntParameterData(const ParameterPtr &param_ptr) {
  std::vector<int> result;
  if (param_ptr == nullptr) {
    MS_LOG(DEBUG) << "Param is nullptr.";
    return result;
  }

  if (!param_ptr->has_default()) {
    MS_LOG(DEBUG) << "Param has not default.";
    return result;
  }
  auto default_param = param_ptr->default_param();
  if (!utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "Tensor info is not tensor::TensorPtr.";
    return result;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  if (default_param_ptr == nullptr) {
    MS_LOG(DEBUG) << "Default param ptr is nullptr.";
    return result;
  }
  if (default_param_ptr->data_type() != kNumberTypeInt32 && default_param_ptr->data_type() != kNumberTypeInt) {
    MS_LOG(DEBUG) << "Default param is not int.";
    return result;
  }

  auto ptr = reinterpret_cast<int *>(default_param_ptr->data_c());
  int shape_size =
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<int>());
  for (int i = 0; i < shape_size; i++) {
    result.emplace_back(ptr[i]);
  }
  return result;
}

bool IsCaseNode(const CNodePtr node) {
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
}  // namespace acl
}  // namespace lite
}  // namespace mindspore
