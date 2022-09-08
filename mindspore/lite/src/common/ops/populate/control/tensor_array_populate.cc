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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_array_parameter.h"

using mindspore::schema::PrimitiveType_TensorArray;
using mindspore::schema::PrimitiveType_TensorArrayRead;
using mindspore::schema::PrimitiveType_TensorArrayWrite;

namespace mindspore {
namespace lite {
OpParameter *PopulateTensorArrayParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_TensorArray();
  MS_CHECK_TRUE_RET(value != nullptr, nullptr);
  MS_CHECK_TRUE_RET(value->element_shape() != nullptr, nullptr);

  auto param = reinterpret_cast<TensorArrayParameter *>(malloc(sizeof(TensorArrayParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorArray nnacl Parameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(TensorArrayParameter));

  param->op_parameter_.type_ = primitive->value_type();
  bool dynamic_size = value->dynamic_size();
  param->dynamic_size_ = dynamic_size;
  bool identical_element_shapes = value->identical_element_shapes();
  param->identical_element_shapes_ = identical_element_shapes;
  std::vector<int> primitive_element_shape(value->element_shape()->begin(), value->element_shape()->end());
  param->element_shape_size_ = static_cast<int>(primitive_element_shape.size());
  auto size = sizeof(int) * static_cast<size_t>(param->element_shape_size_);
  MS_CHECK_LE(size, MAX_SHAPE_SIZE, nullptr);
  memset(param->element_shape_, 0, size);
  memcpy(param->element_shape_, primitive_element_shape.data(), size);
  param->data_type_ = value->data_type();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateTACommonParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);

  auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));

  param->type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_TensorArray, PopulateTensorArrayParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_TensorArrayRead, PopulateTACommonParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_TensorArrayWrite, PopulateTACommonParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
