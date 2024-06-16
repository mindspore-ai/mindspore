/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/gather_d_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "include/registry/converter_context.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
STATUS GetParameterDim(const AnfNodePtr &cnode, int64_t *result_dim) {
  if (result_dim == nullptr) {
    MS_LOG(WARNING) << "result dim is nullptr.";
    return RET_NULL_PTR;
  }

  auto dim_param = cnode->cast<CNodePtr>()->input(THIRD_INPUT)->cast<ParameterPtr>()->default_param();
  if (dim_param == nullptr) {
    MS_LOG(WARNING) << "dim_param is nullptr.";
    return RET_NULL_PTR;
  }
  auto dim_value = std::dynamic_pointer_cast<tensor::Tensor>(dim_param);
  if (dim_value == nullptr) {
    MS_LOG(WARNING) << "dim_value is nullptr.";
    return RET_NULL_PTR;
  }
  if (dim_value->ElementsNum() != 1) {
    MS_LOG(WARNING) << "dim value elements num is not 1, ElementsNum is: " << dim_value->ElementsNum();
    return RET_ERROR;
  }

  if ((dim_value->data_type() != kNumberTypeInt32) && (dim_value->data_type() != kNumberTypeInt64)) {
    MS_LOG(WARNING) << "dim is neither int32 nor int64, now not support other data type.";
    return RET_ERROR;
  }

  if (dim_value->data_type() == kNumberTypeInt32) {
    auto dim_data = static_cast<int32_t *>(dim_value->data_c());
    if (dim_data == nullptr) {
      MS_LOG(WARNING) << "dim_data is nullptr.";
      return RET_ERROR;
    }
    *result_dim = static_cast<int64_t>(dim_data[0]);
  } else if (dim_value->data_type() == kNumberTypeInt64) {
    auto dim_data = static_cast<int64_t *>(dim_value->data_c());
    if (dim_data == nullptr) {
      MS_LOG(WARNING) << "concat_data is nullptr.";
      return RET_ERROR;
    }
    *result_dim = static_cast<int64_t>(dim_data[0]);
  }
  return RET_OK;
}

STATUS ReplaceDimFromInt32ParameterToInt64Value(const CNodePtr &cnode, int64_t dim_val) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);

  auto dim_value_ptr = NewValueNode(MakeValue<int64_t>(dim_val));
  MS_CHECK_TRUE_RET(dim_value_ptr != nullptr, RET_ERROR);

  cnode->set_input(THIRD_INPUT, static_cast<AnfNodePtr>(dim_value_ptr));
  return RET_OK;
}
}  // namespace

// ascend op_adaptor only supports transforming from ValueNode input to ge op attribute, but NOT ParameterNode,
// which lite used to represent const value.
// this mapper replaces the 3rd input('dim') of type ParamterNode to ValueNode-typed input.
STATUS GatherDMapper::Mapper(const CNodePtr &cnode) {
  int64_t dim;

  auto status = GetParameterDim(cnode, &dim);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get parameter 'dim' failed, ret: " << status;
    return status;
  }
  MS_LOG(DEBUG) << "got 'dim' from parameter node, value is " << dim;

  status = ReplaceDimFromInt32ParameterToInt64Value(cnode, dim);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "replace 'dim' of op gather_d from int32 to int64 failed, ret: " << status;
    return status;
  }

  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameGatherD, GatherDMapper)
}  // namespace lite
}  // namespace mindspore
