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
int ExpandDims::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size is invalid";
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output size is invalid";
  }
  auto expand_dims_prim = this->primitive->value_as_ExpandDims();
  int dim = expand_dims_prim->dim();
  if (dim < 0) {
    dim += input->shape().size() + 1;
  }
  if (dim > input->shape().size()) {
    MS_LOG(ERROR) << "attribute dim out of range";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto out_shape = input->shape();
  out_shape.insert(out_shape.begin() + dim, 1, 1);
  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
