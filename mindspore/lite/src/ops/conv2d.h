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

#ifndef LITE_MINDSPORE_LITE_C_OPS_CONV2_D_H_
#define LITE_MINDSPORE_LITE_C_OPS_CONV2_D_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Conv2D : public PrimitiveC {
 public:
  Conv2D() = default;
  ~Conv2D() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Conv2D, PrimitiveC);
  explicit Conv2D(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}

  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) override;
  void SetFormat(int format);
  void SetGroup(int group);
  void SetChannelIn(int channel_in);
  void SetChannelOut(int channel_out);
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

 private:
  void PopulaterConv2DMultiGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group,
                                 const std::vector<AnfNodePtr> &inputs);
  void PopulaterConv2DSingleGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group);
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif

 public:
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int PadUp() const;
  int PadDown() const;
  int PadLeft() const;
  int PadRight() const;

  int GetFormat() const;
  int GetGroup() const;
  int GetChannelIn() const;
  int GetChannelOut() const;
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

 protected:
  void ConvInferShape(int input_h, int input_w, int *output_h, int *output_w);

 protected:
  int pad_u_ = 0;
  int pad_d_ = 0;
  int pad_l_ = 0;
  int pad_r_ = 0;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_CONV2_D_H_
