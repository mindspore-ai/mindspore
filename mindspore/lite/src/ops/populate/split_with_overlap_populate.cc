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
  auto *split_with_over_lap_param =
    reinterpret_cast<SplitWithOverlapParameter *>(malloc(sizeof(SplitWithOverlapParameter)));
  if (split_with_over_lap_param == nullptr) {
    MS_LOG(ERROR) << "malloc PopulateSplitWithOverlapParameter failed.";
    return nullptr;
  }
  memset(split_with_over_lap_param, 0, sizeof(SplitWithOverlapParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SplitWithOverlap();
  split_with_over_lap_param->op_parameter_.type_ = primitive->value_type();

  auto ratio = value->ratio();
  if (ratio->size() > SPLIT_MAX_SLICE_NUM) {
    MS_LOG(ERROR) << "SplitWithOverlap do not support splitting tensor into more than " << SPLIT_MAX_SLICE_NUM
                  << " slices";
    delete split_with_over_lap_param;
    return nullptr;
  }

  split_with_over_lap_param->num_split_ = static_cast<int>(ratio->size());
  split_with_over_lap_param->split_dim_ = value->split_dim();

  auto extend_top = value->extend_top();
  auto extend_bottom = value->extend_bottom();
  if (extend_top->size() != ratio->size() || extend_bottom->size() != ratio->size()) {
    MS_LOG(ERROR) << "The sizes of ratio, extend_top and extend_bottom are not identical";
    delete split_with_over_lap_param;
    return nullptr;
  }

  for (size_t i = 0; i < ratio->size(); ++i) {
    split_with_over_lap_param->ratio_[i] = (*ratio)[i];
    split_with_over_lap_param->extend_top_[i] = (*extend_top)[i];
    split_with_over_lap_param->extend_bottom_[i] = (*extend_bottom)[i];
  }

  split_with_over_lap_param->stride_ = value->stride();
  split_with_over_lap_param->pad_top_ = value->pad_top();

  return reinterpret_cast<OpParameter *>(split_with_over_lap_param);
}
REG_POPULATE(PrimitiveType_SplitWithOverlap, PopulateSplitWithOverlapParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
