/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/dynamic_quant_parameter.h"
using mindspore::schema::PrimitiveType_DynamicQuant;

namespace mindspore {
namespace lite {
OpParameter *PopulateDynamicQuantParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_DynamicQuant();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<DynamicQuantParameter *>(malloc(sizeof(DynamicQuantParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc DynamicQuantParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(DynamicQuantParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->dst_type_ = static_cast<int>(value->dst_type());
  param->symmetric_ = value->symmetric();
  auto prefer_axes = value->prefer_axes();
  if (prefer_axes != nullptr) {
    param->axis_num_ = static_cast<int>(prefer_axes->size());
    if (param->axis_num_ > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "Dynamic quant's prefer_axes's number is more than 8.";
      free(param);
      return nullptr;
    }
    for (int i = 0; i < param->axis_num_; ++i) {
      param->prefer_axes_[i] = prefer_axes->Get(i);
    }
    return reinterpret_cast<OpParameter *>(param);
  }
  auto activation_channel = value->activation_channel();
  if (!activation_channel) {
    return reinterpret_cast<OpParameter *>(param);
  }
  param->axis_num_ = 1;
  param->prefer_axes_[0] = static_cast<int>(value->prefer_axis());
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_DynamicQuant, PopulateDynamicQuantParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
