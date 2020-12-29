/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/split.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/split_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSplitParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *split_param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(split_param, 0, sizeof(SplitParameter));
  auto param = reinterpret_cast<mindspore::lite::Split *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  split_param->op_parameter_.type_ = primitive->Type();
  split_param->num_split_ = param->num_split();
  if (split_param->num_split_ > std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "The value of split_param->num_split_ is too big";
    free(split_param);
    return nullptr;
  }

  /* free split_sizes_ in split op base */
  split_param->split_sizes_ = reinterpret_cast<int *>(malloc(split_param->num_split_ * sizeof(int)));
  if (split_param->split_sizes_ == nullptr) {
    MS_LOG(ERROR) << "malloc split_param split_sizes_ error";
    free(split_param);
    return nullptr;
  }
  memset(split_param->split_sizes_, 0, split_param->num_split_ * sizeof(int));

  auto split_sizes_vector_ = param->size_splits();
  for (size_t i = 0; i < split_sizes_vector_.size(); i++) {
    split_param->split_sizes_[i] = split_sizes_vector_[i];
  }

  split_param->split_dim_ = param->GetSplitDim();
  return reinterpret_cast<OpParameter *>(split_param);
}
Registry SplitParameterRegistry(schema::PrimitiveType_Split, PopulateSplitParameter);
}  // namespace lite
}  // namespace mindspore
