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
int Pooling::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  int input_h = input->shape().at(1);
  int input_w = input->shape().at(2);

  auto pooling_prim = this->primitive->value_as_Pooling();
  MS_ASSERT(pooling_prim != nullptr);
  auto window_h = pooling_prim->windowH();
  auto window_w = pooling_prim->windowW();
  if (pooling_prim->global()) {
    window_h = input_h;
    window_w = input_w;
  }

  int output_h = 0;
  int output_w = 0;
  pad_l_ = pooling_prim->padLeft();
  pad_u_ = pooling_prim->padUp();
  pad_d_ = pooling_prim->padDown();
  pad_r_ = pooling_prim->padRight();
  if (pooling_prim->padMode() == schema::PadMode_SAME) {
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(pooling_prim->strideW()));
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(pooling_prim->strideH()));
    auto pad_h_all = ((output_h - 1) * pooling_prim->strideH() + (window_h - 1) + 1 - input_h);
    auto pad_w_all = ((output_w - 1) * pooling_prim->strideW() + (window_w - 1) + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    auto round_mode = pooling_prim->roundMode();
    if (round_mode == schema::RoundMode_FLOOR) {
      output_h = std::floor(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / pooling_prim->strideH()) + 1;
      output_w = std::floor(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / pooling_prim->strideW()) + 1;
    } else if (round_mode == schema::RoundMode_CEIL) {
      output_h = std::ceil(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / pooling_prim->strideH()) + 1;
      output_w = std::ceil(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / pooling_prim->strideW()) + 1;
    } else {
      MS_LOG(ERROR) << "unsupported round mode.";
    }
  }

  // todo: fmk type
  auto input_shape = input->shape();
  input_shape.at(1) = output_h;
  input_shape.at(2) = output_w;
  output->set_shape(input_shape);
  output->set_data_type(input->data_type());

  // todo: temp fix
  output->SetFormat(schema::Format_NHWC);
  return RET_OK;
}
}  // namespace mindspore::lite
