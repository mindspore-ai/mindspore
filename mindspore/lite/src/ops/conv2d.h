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
  virtual void SetFormat(int format);
  virtual void SetGroup(int group);
  virtual void SetChannelIn(int channel_in);
  virtual void SetChannelOut(int channel_out);
  virtual void SetKernelW(int kernel_w);
  virtual void SetKernelH(int kernel_h);
  virtual void SetStrideW(int stride_w);
  virtual void SetStrideH(int stride_h);
  virtual void SetPadMode(int pad_mode);
  virtual void SetPadUp(int pad_up);
  virtual void SetPadDown(int pad_down);
  virtual void SetPadLeft(int pad_left);
  virtual void SetPadRight(int pad_right);
  virtual void SetDilateW(int dilate_w);
  virtual void SetDilateH(int dilate_h);
  virtual void SetActivationType(int activation_type);

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

  virtual int GetFormat() const;
  virtual int GetGroup() const;
  virtual int GetChannelIn() const;
  virtual int GetChannelOut() const;
  virtual int GetKernelW() const;
  virtual int GetKernelH() const;
  virtual int GetStrideW() const;
  virtual int GetStrideH() const;
  virtual int GetPadMode() const;
  virtual int GetPadUp() const;
  virtual int GetPadDown() const;
  virtual int GetPadLeft() const;
  virtual int GetPadRight() const;
  virtual int GetDilateW() const;
  virtual int GetDilateH() const;
  virtual int GetActivationType() const;

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
