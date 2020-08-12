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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
int Cast::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto cast_prim = this->primitive->value_as_Cast();
  MS_ASSERT(cast_prim != nullptr);
  if (input->data_type() != cast_prim->srcT()) {
    MS_LOG(ERROR) << "input dataType is error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (kSupportDataType.find(input->data_type()) == kSupportDataType.end()) {
    MS_LOG(ERROR) << "Unsupported input data type " << input->data_type();
    return RET_INPUT_TENSOR_ERROR;
  }
  if (cast_prim->dstT() != kNumberTypeFloat && cast_prim->dstT() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid output datatype " << cast_prim->dstT();
    return RET_INPUT_TENSOR_ERROR;
  }
  output->SetFormat(input->GetFormat());
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
