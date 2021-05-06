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
#include "nnacl/constant_of_shape_parameter.h"
using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace mindspore {
namespace lite {
OpParameter *PopulateConstantOfShapeParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_ConstantOfShape();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ConstantOfShapeParameter *>(malloc(sizeof(ConstantOfShapeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConstantOfShapeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConstantOfShapeParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto prim_val = value->value();
  if (prim_val == nullptr) {
    MS_LOG(ERROR) << "val is nullptr";
    free(param);
    return nullptr;
  }
  auto val = std::vector<float>(prim_val->begin(), prim_val->end());
  param->data_type_ = static_cast<int>(value->data_type());
  if (val.empty() || val.size() > 1) {
    MS_LOG(ERROR) << "The value of constant of shape is empty or more than 1.";
  } else {
    switch (param->data_type_) {
      case kNumberTypeFloat32:
        param->value_.f32_value_ = *(prim_val->begin());
        break;
      case kNumberTypeInt32:
        param->value_.int32_value_ = *(prim_val->begin());
        break;
      default:
        MS_LOG(ERROR) << "The value of constant of shape is invalid";
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_ConstantOfShape, PopulateConstantOfShapeParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
