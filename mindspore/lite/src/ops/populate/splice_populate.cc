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
#include "src/ops/populate/populate_register.h"
#include "nnacl/op_base.h"
#include "nnacl/splice_parameter.h"
using mindspore::schema::PrimitiveType_Splice;

namespace mindspore {
namespace lite {
OpParameter *PopulateSpliceParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Splice();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SpliceParameter *>(malloc(sizeof(SpliceParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Splice Parameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SpliceParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto context = value->context();
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    free(param);
    return nullptr;
  }
  std::vector<int> primitive_context(context->begin(), context->end());
  param->context_dim_ = static_cast<int>(primitive_context.size());

  // malloc && memset for context
  param->context_ = reinterpret_cast<int *>(malloc(param->context_dim_ * sizeof(int)));
  if (param->context_ == nullptr) {
    MS_LOG(ERROR) << "malloc param context_ error";
    free(param);
    return nullptr;
  }
  // src_to_dst_row_offset
  int src_to_dst_row_offset = INT32_MIN;
  memset(param->context_, 0, param->context_dim_ * sizeof(int));
  for (int i = 0; i < param->context_dim_; ++i) {
    param->context_[i] = primitive_context.at(i);
    src_to_dst_row_offset = std::max(src_to_dst_row_offset, std::abs(primitive_context.at(i)));
  }

  auto forward_indexes = value->forward_indexes();
  if (forward_indexes == nullptr) {
    MS_LOG(ERROR) << "forward_indexes is nullptr";
    free(param->context_);
    free(param);
    return nullptr;
  }
  std::vector<int> primitive_forward_indexes(forward_indexes->begin(), forward_indexes->end());
  param->forward_indexes_dim_ = static_cast<int>(primitive_forward_indexes.size());

  // malloc && memset for forward_indexes
  param->forward_indexes_ = reinterpret_cast<int *>(malloc(param->forward_indexes_dim_ * sizeof(int)));
  if (param->forward_indexes_ == nullptr) {
    MS_LOG(ERROR) << "malloc param forward_indexes_ error";
    free(param->context_);
    free(param);
    return nullptr;
  }
  memset(param->forward_indexes_, 0, param->forward_indexes_dim_ * sizeof(int));
  for (int i = 0; i < param->context_dim_; ++i) {
    param->context_[i] = primitive_context.at(i);
  }
  param->output_dim_ = value->output_dim();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Splice, PopulateSpliceParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
