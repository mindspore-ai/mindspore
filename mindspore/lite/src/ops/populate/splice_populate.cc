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
  auto *splice_parameter = reinterpret_cast<SpliceParameter *>(malloc(sizeof(SpliceParameter)));
  if (splice_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Splice Parameter failed.";
    return nullptr;
  }
  memset(splice_parameter, 0, sizeof(SpliceParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto splice_primitive = primitive->value_as_Splice();
  if (splice_primitive == nullptr) {
    MS_LOG(ERROR) << "splice_primitive is nullptr";
    return nullptr;
  }
  splice_parameter->op_parameter_.type_ = primitive->value_type();

  auto context = splice_primitive->context();
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    return nullptr;
  }
  std::vector<int> primitive_context(context->begin(), context->end());
  splice_parameter->context_dim_ = static_cast<int>(primitive_context.size());

  // malloc && memset for context
  splice_parameter->context_ = reinterpret_cast<int *>(malloc(splice_parameter->context_dim_ * sizeof(int)));
  if (splice_parameter->context_ == nullptr) {
    MS_LOG(ERROR) << "malloc splice_parameter context_ error";
    free(splice_parameter);
    return nullptr;
  }
  // src_to_dst_row_offset
  int src_to_dst_row_offset = INT32_MIN;
  memset(splice_parameter->context_, 0, splice_parameter->context_dim_ * sizeof(int));
  for (int i = 0; i < splice_parameter->context_dim_; ++i) {
    splice_parameter->context_[i] = primitive_context.at(i);
    src_to_dst_row_offset = std::max(src_to_dst_row_offset, std::abs(primitive_context.at(i)));
  }

  auto forward_indexes = splice_primitive->forward_indexes();
  if (forward_indexes == nullptr) {
    MS_LOG(ERROR) << "forward_indexes is nullptr";
    return nullptr;
  }
  std::vector<int> primitive_forward_indexes(forward_indexes->begin(), forward_indexes->end());
  splice_parameter->forward_indexes_dim_ = static_cast<int>(primitive_forward_indexes.size());

  // malloc && memset for forward_indexes
  splice_parameter->forward_indexes_ =
    reinterpret_cast<int *>(malloc(splice_parameter->forward_indexes_dim_ * sizeof(int)));
  if (splice_parameter->forward_indexes_ == nullptr) {
    MS_LOG(ERROR) << "malloc splice_parameter forward_indexes_ error";
    free(splice_parameter->context_);
    free(splice_parameter);
    return nullptr;
  }
  memset(splice_parameter->forward_indexes_, 0, splice_parameter->forward_indexes_dim_ * sizeof(int));
  for (int i = 0; i < splice_parameter->context_dim_; ++i) {
    splice_parameter->context_[i] = primitive_context.at(i);
  }
  splice_parameter->output_dim_ = splice_primitive->output_dim();
  return reinterpret_cast<OpParameter *>(splice_parameter);
}
REG_POPULATE(PrimitiveType_Splice, PopulateSpliceParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
