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
int DeConv2D::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight = inputs_.at(1);
  MS_ASSERT(weight != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  int32_t input_h = input->Height();
  int32_t input_w = input->Width();

  int32_t output_n = input->Batch();
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_c = weight->Channel();

  auto deconv = GetAttribute();
  int kernel_w = deconv->kernelW();
  int kernel_h = deconv->kernelH();
  int stride_w = deconv->strideW();
  int stride_h = deconv->strideH();
  int dilate_w = deconv->dilateW();
  int dilate_h = deconv->dilateH();
  pad_l_ = deconv->padLeft();
  pad_u_ = deconv->padUp();
  pad_d_ = deconv->padDown();
  pad_r_ = deconv->padRight();
  schema::PadMode pad_mode = deconv->padMode();

  if (pad_mode == schema::PadMode_CAFFE) {
    output_h = (input_h - 1) * stride_h + ((kernel_h - 1) * dilate_h + 1) - pad_u_ - pad_d_;
    output_w = (input_w - 1) * stride_w + ((kernel_w - 1) * dilate_w + 1) - pad_l_ - pad_r_;
  } else if (pad_mode == schema::PadMode_SAME) {
    output_h = input_h * stride_h;
    output_w = input_w * stride_w;
  } else if (pad_mode == schema::PadMode_VALID) {
    output_h = (input_h - 1) * stride_h + kernel_h;
    output_w = (input_w - 1) * stride_w + kernel_w;
  } else {
    MS_LOG(ERROR) << "unsupported pad mode for deconv";
  }

  std::vector<int> out_shape = {output_n, output_h, output_w, output_c};
  output->set_shape(out_shape);
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
