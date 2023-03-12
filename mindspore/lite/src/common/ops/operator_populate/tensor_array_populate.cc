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
#include "nnacl/tensor_array_parameter.h"
#include "ops/tensor_array.h"
#include "ops/tensor_array_read.h"
#include "ops/tensor_array_write.h"
using mindspore::ops::kNameTensorArray;
using mindspore::ops::kNameTensorArrayRead;
using mindspore::ops::kNameTensorArrayWrite;
using mindspore::schema::PrimitiveType_TensorArray;
namespace mindspore {
namespace lite {
OpParameter *PopulateTensorArrayOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<TensorArrayParameter *>(PopulateOpParameter<TensorArrayParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new TensorArrayParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::TensorArray *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TensorArray.";
    free(param);
    return nullptr;
  }
  param->dynamic_size_ = op->get_dynamic_size();
  param->identical_element_shapes_ = op->get_identical_element_shapes();
  param->element_shape_size_ = static_cast<int>(op->get_element_shape().size());
  auto size = sizeof(int) * static_cast<size_t>(param->element_shape_size_);
  MS_CHECK_LE(size, MAX_SHAPE_SIZE, nullptr);
  memset(param->element_shape_, 0, size);
  memcpy(param->element_shape_, op->get_element_shape().data(), size);
  param->data_type_ = op->get_data_type();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameTensorArray, PrimitiveType_TensorArray, PopulateTensorArrayOpParameter)
}  // namespace lite
}  // namespace mindspore
