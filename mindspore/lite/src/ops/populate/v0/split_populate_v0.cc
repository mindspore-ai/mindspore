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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/split_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSplitParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto split_prim = primitive->value_as_Split();
  auto *split_param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(split_param, 0, sizeof(SplitParameter));
  split_param->op_parameter_.type_ = schema::PrimitiveType_Split;
  split_param->num_split_ = split_prim->numberSplit();
  if (split_param->num_split_ > std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "The value of split_param->num_split_ is too big";
    free(split_param);
    return nullptr;
  }
  int *split_sizes = reinterpret_cast<int *>(malloc(split_param->num_split_ * sizeof(int)));
  if (split_sizes == nullptr) {
    MS_LOG(ERROR) << "malloc split size of SplitParameter failed.";
    free(split_param);
    return nullptr;
  }
  memset(split_sizes, 0, split_param->num_split_ * sizeof(int));
  split_param->split_sizes_ = split_sizes;
  auto split_sizes_vector_ = split_prim->sizeSplits();
  if (split_sizes_vector_ != NULL) {
    int i = 0;
    for (auto iter = split_sizes_vector_->begin(); iter != split_sizes_vector_->end(); iter++) {
      split_param->split_sizes_[i++] = *iter;
    }
    split_param->split_count_ = split_param->num_split_;
  } else {
    split_param->split_count_ = 0;
  }
  split_param->split_dim_ = split_prim->splitDim();
  return reinterpret_cast<OpParameter *>(split_param);
}
}  // namespace

Registry g_splitV0ParameterRegistry(schema::v0::PrimitiveType_Split, PopulateSplitParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
