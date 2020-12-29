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

#include "src/ops/constant_of_shape.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/constant_of_shape.h"

namespace mindspore::lite {
namespace {
OpParameter *PopulateConstantOfShapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto attr =
    reinterpret_cast<mindspore::lite::ConstantOfShape *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ConstantOfShapeParameter *param =
    reinterpret_cast<ConstantOfShapeParameter *>(malloc(sizeof(ConstantOfShapeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConstantOfShapeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConstantOfShapeParameter));
  param->op_parameter_.type_ = primitive->Type();
  param->data_type_ = attr->GetDataType();
  auto value = attr->GetValue();
  if (value.empty() || value.size() > 1) {
    MS_LOG(ERROR) << "The value of constant of shape is empty or more than 1.";
  } else {
    switch (param->data_type_) {
      case kNumberTypeFloat32:
        param->value_.f32_value_ = attr->GetValue().at(0);
        break;
      case kNumberTypeInt32:
        param->value_.int32_value_ = attr->GetValue().at(0);
        break;
      default:
        MS_LOG(ERROR) << "The value of constant of shape is invalid";
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}
Registry ConstantOfShapeParameterRegistry(schema::PrimitiveType_ConstantOfShape, PopulateConstantOfShapeParameter);
}  // namespace
}  // namespace mindspore::lite
