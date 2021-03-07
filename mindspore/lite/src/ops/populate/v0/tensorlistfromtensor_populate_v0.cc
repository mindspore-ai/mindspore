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
#include "nnacl/tensorlist_parameter.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateTensorListFromTensorParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto tensorList = primitive->value_as_TensorListFromTensor();
  TensorListParameter *TensorList_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (TensorList_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(TensorList_param, 0, sizeof(TensorListParameter));
  TensorList_param->op_parameter_.type_ = schema::PrimitiveType_TensorListFromTensor;
  TensorList_param->shape_type_ = (TypeId)(tensorList->shapeType());
  TensorList_param->element_dtype_ = (TypeId)(tensorList->elementDType());
  return reinterpret_cast<OpParameter *>(TensorList_param);
}
}  // namespace

Registry g_tensorListFromTensorV0ParameterRegistry(schema::v0::PrimitiveType_TensorListFromTensor,
                                                   PopulateTensorListFromTensorParameter, SCHEMA_V0);

}  // namespace lite
}  // namespace mindspore
