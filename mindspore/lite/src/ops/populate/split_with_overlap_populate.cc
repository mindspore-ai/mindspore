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
using mindspore::schema::PrimitiveType_SplitWithOverlap;

namespace mindspore {
namespace lite {
OpParameter *PopulateSplitWithOverlapParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SplitWithOverlap();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SplitWithOverlapParameter *>(malloc(sizeof(SplitWithOverlapParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc PopulateSplitWithOverlapParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SplitWithOverlapParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->num_split_ = value->number_split();
  param->split_dim_ = value->split_dim();

  if (param->num_split_ > SPLIT_MAX_SLICE_NUM) {
    MS_LOG(ERROR) << "SplitWithOverlap num_split_ error.";
    free(param);
    return nullptr;
  }

  auto ratio = value->ratio();
  auto extend_top = value->extend_top();
  auto extend_bottom = value->extend_bottom();
  if (ratio == nullptr || extend_top == nullptr || extend_bottom == nullptr) {
    MS_LOG(ERROR) << "SplitWithOverlap parameter is nullptr.";
    free(param);
    return nullptr;
  }
  if (static_cast<int>(ratio->size()) != param->num_split_ ||
      static_cast<int>(extend_top->size()) != param->num_split_ ||
      static_cast<int>(extend_bottom->size()) != param->num_split_) {
    MS_LOG(ERROR) << "SplitWithOverlap parameter size error.";
    free(param);
    return nullptr;
  }

  for (size_t i = 0; i < ratio->size(); ++i) {
    param->ratio_[i] = (*ratio)[i];
    param->extend_top_[i] = (*extend_top)[i];
    param->extend_bottom_[i] = (*extend_bottom)[i];
  }

  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_SplitWithOverlap, PopulateSplitWithOverlapParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
