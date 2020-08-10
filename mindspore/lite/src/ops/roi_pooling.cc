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
int ROIPooling::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kDoubleNum;
    return RET_ERROR;
  }
  auto input = inputs_.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto roi = inputs_.at(1);
  if (roi == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = outputs_.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  auto ROIPooling = GetAttribute();
  auto new_h = ROIPooling->pooledH();
  auto new_w = ROIPooling->pooledW();

  auto shape_data = roi->shape();

  std::vector<int> output_shape;
  output_shape.push_back(shape_data[0]);
  output_shape.push_back(new_h);
  output_shape.push_back(new_w);
  output_shape.push_back(input->Channel());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
