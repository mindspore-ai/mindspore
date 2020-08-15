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

#include "c_ops/pooling.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Pooling::GetFormat() const { return this->primitive->value.AsPooling()->format; }
int Pooling::GetPoolingMode() const { return this->primitive->value.AsPooling()->poolingMode; }
bool Pooling::GetGlobal() const { return this->primitive->value.AsPooling()->global; }
int Pooling::GetWindowW() const { return this->primitive->value.AsPooling()->windowW; }
int Pooling::GetWindowH() const { return this->primitive->value.AsPooling()->windowH; }
int Pooling::GetStrideW() const { return this->primitive->value.AsPooling()->strideW; }
int Pooling::GetStrideH() const { return this->primitive->value.AsPooling()->strideH; }
int Pooling::GetPadMode() const { return this->primitive->value.AsPooling()->padMode; }
int Pooling::GetPadUp() const { return this->primitive->value.AsPooling()->padUp; }
int Pooling::GetPadDown() const { return this->primitive->value.AsPooling()->padDown; }
int Pooling::GetPadLeft() const { return this->primitive->value.AsPooling()->padLeft; }
int Pooling::GetPadRight() const { return this->primitive->value.AsPooling()->padRight; }
int Pooling::GetRoundMode() const { return this->primitive->value.AsPooling()->roundMode; }

void Pooling::SetFormat(int format) { this->primitive->value.AsPooling()->format = (schema::Format)format; }
void Pooling::SetPoolingMode(int pooling_mode) {
  this->primitive->value.AsPooling()->poolingMode = (schema::PoolMode)pooling_mode;
}
void Pooling::SetGlobal(bool global) { this->primitive->value.AsPooling()->global = global; }
void Pooling::SetWindowW(int window_w) { this->primitive->value.AsPooling()->windowW = window_w; }
void Pooling::SetWindowH(int window_h) { this->primitive->value.AsPooling()->windowH = window_h; }
void Pooling::SetStrideW(int stride_w) { this->primitive->value.AsPooling()->strideW = stride_w; }
void Pooling::SetStrideH(int stride_h) { this->primitive->value.AsPooling()->strideH = stride_h; }
void Pooling::SetPadMode(int pad_mode) { this->primitive->value.AsPooling()->padMode = (schema::PadMode)pad_mode; }
void Pooling::SetPadUp(int pad_up) { this->primitive->value.AsPooling()->padUp = pad_up; }
void Pooling::SetPadDown(int pad_down) { this->primitive->value.AsPooling()->padDown = pad_down; }
void Pooling::SetPadLeft(int pad_left) { this->primitive->value.AsPooling()->padLeft = pad_left; }
void Pooling::SetPadRight(int pad_right) { this->primitive->value.AsPooling()->padRight = pad_right; }
void Pooling::SetRoundMode(int round_mode) {
  this->primitive->value.AsPooling()->roundMode = (schema::RoundMode)round_mode;
}

#else

int Pooling::GetFormat() const { return this->primitive->value_as_Pooling()->format(); }
int Pooling::GetPoolingMode() const { return this->primitive->value_as_Pooling()->poolingMode(); }
bool Pooling::GetGlobal() const { return this->primitive->value_as_Pooling()->global(); }
int Pooling::GetWindowW() const { return this->primitive->value_as_Pooling()->windowW(); }
int Pooling::GetWindowH() const { return this->primitive->value_as_Pooling()->windowH(); }
int Pooling::GetStrideW() const { return this->primitive->value_as_Pooling()->strideW(); }
int Pooling::GetStrideH() const { return this->primitive->value_as_Pooling()->strideH(); }
int Pooling::GetPadMode() const { return this->primitive->value_as_Pooling()->padMode(); }
int Pooling::GetPadUp() const { return this->primitive->value_as_Pooling()->padUp(); }
int Pooling::GetPadDown() const { return this->primitive->value_as_Pooling()->padDown(); }
int Pooling::GetPadLeft() const { return this->primitive->value_as_Pooling()->padLeft(); }
int Pooling::GetPadRight() const { return this->primitive->value_as_Pooling()->padRight(); }
int Pooling::GetRoundMode() const { return this->primitive->value_as_Pooling()->roundMode(); }

void Pooling::SetFormat(int format) {}
void Pooling::SetPoolingMode(int pooling_mode) {}
void Pooling::SetGlobal(bool global) {}
void Pooling::SetWindowW(int window_w) {}
void Pooling::SetWindowH(int window_h) {}
void Pooling::SetStrideW(int stride_w) {}
void Pooling::SetStrideH(int stride_h) {}
void Pooling::SetPadMode(int pad_mode) {}
void Pooling::SetPadUp(int pad_up) {}
void Pooling::SetPadDown(int pad_down) {}
void Pooling::SetPadLeft(int pad_left) {}
void Pooling::SetPadRight(int pad_right) {}
void Pooling::SetRoundMode(int round_mode) {}
#endif
int Pooling::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  int input_h = input->shape().at(1);
  int input_w = input->shape().at(2);

  MS_ASSERT(pooling_prim != nullptr);
  auto window_h = GetWindowH();
  auto window_w = GetWindowH();
  if (GetGlobal()) {
    window_h = input_h;
    window_w = input_w;
  }

  int output_h = 0;
  int output_w = 0;
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();
  if ((schema::PadMode)GetPadMode() == schema::PadMode_SAME) {
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(GetStrideW()));
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(GetStrideH()));
    auto pad_h_all = ((output_h - 1) * GetStrideH() + (window_h - 1) + 1 - input_h);
    auto pad_w_all = ((output_w - 1) * GetStrideW() + (window_w - 1) + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    auto round_mode = GetRoundMode();
    if (round_mode == schema::RoundMode_FLOOR) {
      output_h = std::floor(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / GetStrideH()) + 1;
      output_w = std::floor(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / GetStrideW()) + 1;
    } else if (round_mode == schema::RoundMode_CEIL) {
      output_h = std::ceil(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / GetStrideH()) + 1;
      output_w = std::ceil(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / GetStrideW()) + 1;
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
  return 0;
}
}  // namespace mindspore
