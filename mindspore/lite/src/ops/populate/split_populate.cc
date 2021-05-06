/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/split_parameter.h"
using mindspore::schema::PrimitiveType_Split;

namespace mindspore {
namespace lite {
OpParameter *PopulateSplitParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Split();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SplitParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->num_split_ = value->output_num();
  if (param->num_split_ > std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "The value of param->num_split_ is too big";
    free(param);
    return nullptr;
  }

  /* free split_sizes_ in split op base */
  param->split_sizes_ = reinterpret_cast<int *>(malloc(param->num_split_ * sizeof(int)));
  if (param->split_sizes_ == nullptr) {
    MS_LOG(ERROR) << "malloc param split_sizes_ error";
    free(param);
    return nullptr;
  }
  memset(param->split_sizes_, 0, param->num_split_ * sizeof(int));
  auto split_sizes_vector_ = value->size_splits();
  if (split_sizes_vector_ != nullptr) {
    int i = 0;
    for (auto iter : *split_sizes_vector_) {
      param->split_sizes_[i++] = iter;
    }
    param->split_count_ = param->num_split_;
  } else {
    param->split_count_ = 0;
  }
  param->split_dim_ = value->axis();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Split, PopulateSplitParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
