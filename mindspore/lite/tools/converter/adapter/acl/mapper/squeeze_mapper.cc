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

#include "tools/converter/adapter/acl/mapper/squeeze_mapper.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "transform/graph_ir/op_declare/array_ops_declare.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
bool SqueezeMapper::GetAxisValue(AnfNodePtr input_node, std::vector<int64_t> *axis_val) {
  if (input_node == nullptr || axis_val == nullptr) {
    return false;
  }
  ValuePtr value = nullptr;
  if (input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
    auto value_node = input_node->cast<ValueNodePtr>();
    if (value_node) {
      value = value_node->value();
    }
  } else if (input_node->isa<Parameter>()) {
    auto parameter = input_node->cast<ParameterPtr>();
    if (parameter->has_default()) {
      value = parameter->default_param();
    }
  }
  if (value == nullptr) {
    return false;
  }
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor == nullptr || tensor->data().const_data() == nullptr) {
      return false;
    }
    if (tensor->data_type() == kNumberTypeInt64) {
      auto int64_val = reinterpret_cast<const int64_t *>(tensor->data_c());
      *axis_val = std::vector<int64_t>(int64_val, int64_val + tensor->ElementsNum());
    } else if (tensor->data_type() == kNumberTypeInt32) {
      auto int32_val = reinterpret_cast<const int32_t *>(tensor->data_c());
      *axis_val = std::vector<int64_t>(int32_val, int32_val + tensor->ElementsNum());
    } else {
      MS_LOG(ERROR) << "Failed to get axis val from tensor, data type: " << tensor->data_type()
                    << ", axis value node: " << input_node->fullname_with_scope();
      return false;
    }
    return true;
  }
  if (value->isa<Int32Imm>()) {
    *axis_val = std::vector<int64_t>{value->cast<Int32ImmPtr>()->value()};
  } else if (value->isa<Int64Imm>()) {
    *axis_val = std::vector<int64_t>{value->cast<Int64ImmPtr>()->value()};
  } else {
    MS_LOG(ERROR) << "Failed to get axis val, value type: " << value->type_name()
                  << ", axis value node: " << input_node->fullname_with_scope();
    return false;
  }
  return true;
}

STATUS SqueezeMapper::Mapper(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  constexpr size_t input_count_with_const = 2 + 1;
  constexpr size_t axis_input_index = 2;
  if (cnode->size() != input_count_with_const) {
    return RET_OK;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto axis_input = cnode->input(axis_input_index);
  std::vector<int64_t> axis_val;
  if (GetAxisValue(axis_input, &axis_val)) {
    src_prim->AddAttr(ops::kAxis, MakeValue(axis_val));
    return lite::RET_OK;
  }
  PrimitivePtr dst_prim = std::make_shared<acl::SqueezeV3>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "Failed to make SqueezeV3 primitive.";
    return lite::RET_ERROR;
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameSqueeze, SqueezeMapper)
}  // namespace lite
}  // namespace mindspore
