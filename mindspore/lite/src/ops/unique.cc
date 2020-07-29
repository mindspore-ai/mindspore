/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
int Unique::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto &input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto &output0 = outputs_.at(0);
  MS_ASSERT(output0 != nullptr);
  auto &output1 = outputs_.at(1);
  MS_ASSERT(output1 != nullptr);
  output0->set_shape(input->shape());
  output0->set_data_type(input->data_type());
  output1->set_shape(input->shape());
  output1->set_data_type(kNumberTypeInt32);
  output1->SetFormat(input->GetFormat());
  output0->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
