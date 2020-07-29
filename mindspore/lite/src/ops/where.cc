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
int Where::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "where input or output number invalid, Input size:" << inputs_.size()
                  << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  if (inputs_.size() < 3) {
    MS_LOG(ERROR) << "Input shape tensors should b";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input0 = inputs_.at(0);
  auto input1 = inputs_.at(1);
  auto input2 = inputs_.at(2);
  int num = input0->ElementsNum();
  int num1 = input1->ElementsNum();
  int num2 = input2->ElementsNum();
  int nummax = num > num1 ? num : (num1 > num2 ? num1 : num2);

  auto shape_tmp = inputs_.at(0)->shape();
  auto shape_tmp1 = inputs_.at(1)->shape();
  auto shape_tmp2 = inputs_.at(2)->shape();
  int axisout = 0;
  int temp = 0;
  for (int j = 0; j < shape_tmp.size(); j++) {
    if (shape_tmp[j] == shape_tmp1[j] && shape_tmp[j] != shape_tmp2[j]) {
      axisout = j;
      break;
    }
    if (shape_tmp[j] == shape_tmp2[j] && shape_tmp[j] != shape_tmp1[j]) {
      axisout = j;
      break;
    }
    if (shape_tmp1[j] == shape_tmp2[j] && shape_tmp[j] != shape_tmp1[j]) {
      axisout = j;
      break;
    }
    temp += 1;
    if (temp == shape_tmp.size()) {
      outputs_[0]->set_shape(shape_tmp);
      output->set_data_type(input->data_type());
      return RET_OK;
    }
  }

  auto output_shape = shape_tmp;
  output_shape[axisout] = nummax;
  outputs_[0]->set_shape(output_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
