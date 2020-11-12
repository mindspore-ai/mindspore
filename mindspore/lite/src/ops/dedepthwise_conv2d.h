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

#ifndef MINDSPORE_LITE_SRC_OPS_DEDEPTHWISE_CONV2D_H_
#define MINDSPORE_LITE_SRC_OPS_DEDEPTHWISE_CONV2D_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class DeDepthwiseConv2D : public PrimitiveC {
 public:
  DeDepthwiseConv2D() = default;
  ~DeDepthwiseConv2D() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(DeDepthwiseConv2D, PrimitiveC);
  explicit DeDepthwiseConv2D(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetFormat(int format);
  void SetChannelIn(int channel_in);
  void SetChannelMultiplier(int channel_multiplier);
  void SetKernelW(int kernel_w);
  void SetKernelH(int kernel_h);
  void SetStrideW(int stride_w);
  void SetStrideH(int stride_h);
  void SetPadMode(int pad_mode);
  void SetPadUp(int pad_up);
  void SetPadDown(int pad_down);
  void SetPadLeft(int pad_left);
  void SetPadRight(int pad_right);
  void SetDilateW(int dilate_w);
  void SetDilateH(int dilate_h);
  void SetHasBias(bool has_bias);
  void SetActivationType(int activation_type);
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int GetFormat() const;
  int GetChannelIn() const;
  int GetChannelMultiplier() const;
  int GetKernelW() const;
  int GetKernelH() const;
  int GetStrideW() const;
  int GetStrideH() const;
  int GetPadMode() const;
  int GetPadUp() const;
  int GetPadDown() const;
  int GetPadLeft() const;
  int GetPadRight() const;
  int GetDilateW() const;
  int GetDilateH() const;
  bool GetHasBias() const;
  int GetActivationType() const;

  int PadUp() const { return this->pad_u_; }
  int PadDown() const { return this->pad_d_; }
  int PadLeft() const { return this->pad_l_; }
  int PadRight() const { return this->pad_r_; }

 protected:
  int pad_u_ = 0;
  int pad_d_ = 0;
  int pad_l_ = 0;
  int pad_r_ = 0;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_OPS_DEDEPTHWISE_CONV2D_H_
