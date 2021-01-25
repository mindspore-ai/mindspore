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

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSplitParameter(const void *prim) {
  auto *split_param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(split_param, 0, sizeof(SplitParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Split();
  split_param->op_parameter_.type_ = primitive->value_type();
  split_param->num_split_ = value->output_num();
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
  auto split_sizes_vector_ = value->size_splits();
  if (split_sizes_vector_ != NULL) {
    int i = 0;
    for (auto iter : *split_sizes_vector_) {
      split_param->split_sizes_[i++] = iter;
    }
    split_param->split_count_ = split_param->num_split_;
  } else {
    split_param->split_count_ = 0;
  }
  split_param->split_dim_ = value->axis();
  return reinterpret_cast<OpParameter *>(split_param);
}
}  // namespace
Registry g_splitParameterRegistry(schema::PrimitiveType_Split, PopulateSplitParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
