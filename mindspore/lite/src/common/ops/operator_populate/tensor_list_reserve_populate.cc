/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/tensorlist_parameter.h"
#include "ops/tensor_list_reserve.h"
using mindspore::ops::kNameTensorListReserve;
using mindspore::schema::PrimitiveType_TensorListReserve;
namespace mindspore {
namespace lite {
OpParameter *PopulateTensorListReserveOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<TensorListParameter *>(PopulateOpParameter<TensorListParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new TensorListParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::TensorListReserve *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TensorListReserve.";
    free(param);
    return nullptr;
  }
  param->element_dtype_ = static_cast<int>(op->get_element_dtype());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameTensorListReserve, PrimitiveType_TensorListReserve, PopulateTensorListReserveOpParameter)
}  // namespace lite
}  // namespace mindspore
