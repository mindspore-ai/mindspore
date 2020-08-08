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
void Conv2D::ConvInferShape(int input_h, int input_w, int *output_h, int *output_w) {
  MS_ASSERT(this->primitive != nullptr);
  auto conv2DPrim = this->primitive->value_as_Conv2D();
  int kernel_w = conv2DPrim->kernelW();
  int kernel_h = conv2DPrim->kernelH();
  int stride_w = conv2DPrim->strideW();
  int stride_h = conv2DPrim->strideH();
  int dilate_w = conv2DPrim->dilateW();
  int dilate_h = conv2DPrim->dilateH();
  pad_l_ = conv2DPrim->padLeft();
  pad_u_ = conv2DPrim->padUp();
  pad_d_ = conv2DPrim->padDown();
  pad_r_ = conv2DPrim->padRight();

  if (conv2DPrim->padMode() == schema::PadMode_SAME) {
    *output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w));
    *output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h));
    auto pad_h_all = ((*output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    auto pad_w_all = ((*output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    *output_w = std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ -
        (static_cast<float>(kernel_w) - 1) * static_cast<float>(dilate_w)) / static_cast<float>(stride_w));
    *output_h = std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ -
        (static_cast<float>(kernel_h) - 1) * static_cast<float>(dilate_h)) / static_cast<float>(stride_h));
  }
}

int Conv2D::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != 2 && inputs_.size() != 3) {
    MS_LOG(ERROR) << "Add should has two or three inputs";
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "Add should has one outputs";
    return RET_ERROR;
  }
  auto *input_tensor = inputs_.front();
  auto *weight_tensor = inputs_.at(1);
  auto *out_tensor = outputs_.front();
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(out_tensor != nullptr);

  auto in_shape = input_tensor->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int output_w = 0, output_h = 0;

  this->ConvInferShape(input_h, input_w, &output_h, &output_w);

  std::vector<int> out_shape{input_tensor->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  out_shape.at(3) = weight_tensor->shape()[0];
  out_tensor->set_shape(out_shape);
  out_tensor->SetFormat(input_tensor->GetFormat());
  out_tensor->set_data_type(input_tensor->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite

