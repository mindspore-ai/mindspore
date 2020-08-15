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

#include "c_ops/pooling_grad.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int PoolingGrad::GetFormat() const { return this->primitive->value.AsPoolingGrad()->format; }
int PoolingGrad::GetPoolingMode() const { return this->primitive->value.AsPoolingGrad()->poolingMode; }
bool PoolingGrad::GetGlobal() const { return this->primitive->value.AsPoolingGrad()->global; }
int PoolingGrad::GetWindowW() const { return this->primitive->value.AsPoolingGrad()->windowW; }
int PoolingGrad::GetWindowH() const { return this->primitive->value.AsPoolingGrad()->windowH; }
int PoolingGrad::GetStrideW() const { return this->primitive->value.AsPoolingGrad()->strideW; }
int PoolingGrad::GetStrideH() const { return this->primitive->value.AsPoolingGrad()->strideH; }
int PoolingGrad::GetPadMode() const { return this->primitive->value.AsPoolingGrad()->padMode; }
int PoolingGrad::GetPadUp() const { return this->primitive->value.AsPoolingGrad()->padUp; }
int PoolingGrad::GetPadDown() const { return this->primitive->value.AsPoolingGrad()->padDown; }
int PoolingGrad::GetPadLeft() const { return this->primitive->value.AsPoolingGrad()->padLeft; }
int PoolingGrad::GetPadRight() const { return this->primitive->value.AsPoolingGrad()->padRight; }
int PoolingGrad::GetRoundMode() const { return this->primitive->value.AsPoolingGrad()->roundMode; }

void PoolingGrad::SetFormat(int format) { this->primitive->value.AsPoolingGrad()->format = (schema::Format)format; }
void PoolingGrad::SetPoolingMode(int pooling_mode) {
  this->primitive->value.AsPoolingGrad()->poolingMode = (schema::PoolMode)pooling_mode;
}
void PoolingGrad::SetGlobal(bool global) { this->primitive->value.AsPoolingGrad()->global = global; }
void PoolingGrad::SetWindowW(int window_w) { this->primitive->value.AsPoolingGrad()->windowW = window_w; }
void PoolingGrad::SetWindowH(int window_h) { this->primitive->value.AsPoolingGrad()->windowH = window_h; }
void PoolingGrad::SetStrideW(int stride_w) { this->primitive->value.AsPoolingGrad()->strideW = stride_w; }
void PoolingGrad::SetStrideH(int stride_h) { this->primitive->value.AsPoolingGrad()->strideH = stride_h; }
void PoolingGrad::SetPadMode(int pad_mode) {
  this->primitive->value.AsPoolingGrad()->padMode = (schema::PadMode)pad_mode;
}
void PoolingGrad::SetPadUp(int pad_up) { this->primitive->value.AsPoolingGrad()->padUp = pad_up; }
void PoolingGrad::SetPadDown(int pad_down) { this->primitive->value.AsPoolingGrad()->padDown = pad_down; }
void PoolingGrad::SetPadLeft(int pad_left) { this->primitive->value.AsPoolingGrad()->padLeft = pad_left; }
void PoolingGrad::SetPadRight(int pad_right) { this->primitive->value.AsPoolingGrad()->padRight = pad_right; }
void PoolingGrad::SetRoundMode(int round_mode) {
  this->primitive->value.AsPoolingGrad()->roundMode = (schema::RoundMode)round_mode;
}

#else

int PoolingGrad::GetFormat() const { return this->primitive->value_as_PoolingGrad()->format(); }
int PoolingGrad::GetPoolingMode() const { return this->primitive->value_as_PoolingGrad()->poolingMode(); }
bool PoolingGrad::GetGlobal() const { return this->primitive->value_as_PoolingGrad()->global(); }
int PoolingGrad::GetWindowW() const { return this->primitive->value_as_PoolingGrad()->windowW(); }
int PoolingGrad::GetWindowH() const { return this->primitive->value_as_PoolingGrad()->windowH(); }
int PoolingGrad::GetStrideW() const { return this->primitive->value_as_PoolingGrad()->strideW(); }
int PoolingGrad::GetStrideH() const { return this->primitive->value_as_PoolingGrad()->strideH(); }
int PoolingGrad::GetPadMode() const { return this->primitive->value_as_PoolingGrad()->padMode(); }
int PoolingGrad::GetPadUp() const { return this->primitive->value_as_PoolingGrad()->padUp(); }
int PoolingGrad::GetPadDown() const { return this->primitive->value_as_PoolingGrad()->padDown(); }
int PoolingGrad::GetPadLeft() const { return this->primitive->value_as_PoolingGrad()->padLeft(); }
int PoolingGrad::GetPadRight() const { return this->primitive->value_as_PoolingGrad()->padRight(); }
int PoolingGrad::GetRoundMode() const { return this->primitive->value_as_PoolingGrad()->roundMode(); }

void PoolingGrad::SetFormat(int format) {}
void PoolingGrad::SetPoolingMode(int pooling_mode) {}
void PoolingGrad::SetGlobal(bool global) {}
void PoolingGrad::SetWindowW(int window_w) {}
void PoolingGrad::SetWindowH(int window_h) {}
void PoolingGrad::SetStrideW(int stride_w) {}
void PoolingGrad::SetStrideH(int stride_h) {}
void PoolingGrad::SetPadMode(int pad_mode) {}
void PoolingGrad::SetPadUp(int pad_up) {}
void PoolingGrad::SetPadDown(int pad_down) {}
void PoolingGrad::SetPadLeft(int pad_left) {}
void PoolingGrad::SetPadRight(int pad_right) {}
void PoolingGrad::SetRoundMode(int round_mode) {}
#endif
}  // namespace mindspore
