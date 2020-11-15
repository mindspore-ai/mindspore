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
  SplitParameter *split_param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(split_param, 0, sizeof(SplitParameter));
  auto param = reinterpret_cast<mindspore::lite::Split *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  split_param->op_parameter_.type_ = primitive->Type();
  split_param->num_split_ = param->GetNumberSplit();
  int *split_sizes = reinterpret_cast<int *>(malloc(split_param->num_split_ * sizeof(int)));
  if (split_sizes == nullptr) {
    MS_LOG(ERROR) << "malloc split size of SplitParameter failed.";
    return nullptr;
  }
  memset(split_sizes, 0, split_param->num_split_ * sizeof(int));
  split_param->split_sizes_ = split_sizes;
  auto split_sizes_vector_ = param->GetSizeSplits();
  int i = 0;
  for (auto iter = split_sizes_vector_.begin(); iter != split_sizes_vector_.end(); iter++) {
    split_param->split_sizes_[i++] = *iter;
  }
  split_param->split_dim_ = param->GetSplitDim();
  return reinterpret_cast<OpParameter *>(split_param);
}
Registry SplitParameterRegistry(schema::PrimitiveType_Split, PopulateSplitParameter);
}  // namespace lite
}  // namespace mindspore
