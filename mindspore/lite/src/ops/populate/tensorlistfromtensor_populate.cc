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

#include "nnacl/tensorlist_parameter.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "src/ops/tensorlist_fromtensor.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateTensorListFromTensorParameter(const mindspore::lite::PrimitiveC *primitive) {
  TensorListParameter *TensorList_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (TensorList_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(TensorList_param, 0, sizeof(TensorListParameter));
  TensorList_param->op_parameter_.type_ = primitive->Type();
  auto tensorList =
    reinterpret_cast<mindspore::lite::TensorListFromTensor *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  TensorList_param->shape_type_ = (TypeId)(tensorList->GetShapeType());
  TensorList_param->element_dtype_ = (TypeId)(tensorList->GetElementDType());
  return reinterpret_cast<OpParameter *>(TensorList_param);
}
Registry TensorListFromTensorParameterRegistry(schema::PrimitiveType_TensorListFromTensor,
                                               PopulateTensorListFromTensorParameter);

}  // namespace lite
}  // namespace mindspore
