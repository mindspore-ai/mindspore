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

#ifndef MINDSPORE_LITE_SRC_OPS_OPS_H_
#define MINDSPORE_LITE_SRC_OPS_OPS_H_

#include <vector>
#include <set>
#include <cmath>
#include "schema/model_generated.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace lite::tensor {
class Tensor;
}
namespace lite {
constexpr uint32_t kSingleNum = 1;
constexpr uint32_t kDoubleNum = 2;
constexpr uint32_t kMultiNum = 3;
constexpr uint32_t kNHWC_n_index = 0;
constexpr uint32_t kNHWC_h_index = 1;
constexpr uint32_t kNHWC_w_index = 2;
constexpr uint32_t kNHWC_c_index = 3;
constexpr uint32_t kDimension_4d = 4;

const std::set<int> kSupportDataType = {kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeFloat32};

class Primitive {
 public:
  explicit Primitive(schema::Primitive *primitive) : primitive(primitive) {}
  static Primitive *CreatePrimitive(schema::Primitive *primitive);
  virtual ~Primitive() {}
  const schema::Primitive *Value() const { return this->primitive; }
  const bool GetInferFlag() const { return this->infer_flag_; }
  void SetInferFlag(bool flag) { this->infer_flag_ = flag; }
  schema::PrimitiveType Type() const { return this->primitive->value_type(); }
  const void *Attribute() const { return this->primitive->value(); }
  virtual int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_);

 protected:
  schema::Primitive *primitive;
  bool infer_flag_ = true;
};

class ROIPooling : public Primitive {
 public:
  explicit ROIPooling(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ROIPooling *GetAttribute() const { return this->primitive->value_as_ROIPooling(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Conv2D : public Primitive {
 public:
  explicit Conv2D(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Conv2D *GetAttribute() const { return this->primitive->value_as_Conv2D(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
  int PadUp() const { return this->pad_u_; }
  int PadDown() const { return this->pad_d_; }
  int PadLeft() const { return this->pad_l_; }
  int PadRight() const { return this->pad_r_; }

 protected:
  void ConvInferShape(int input_h, int input_w, int *output_h, int *output_w);

 protected:
  int pad_u_ = 0;
  int pad_d_ = 0;
  int pad_l_ = 0;
  int pad_r_ = 0;
};

class Pooling : public Primitive {
 public:
  explicit Pooling(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Pooling *GetAttribute() const { return this->primitive->value_as_Pooling(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
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

class BatchNorm : public Primitive {
 public:
  explicit BatchNorm(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::BatchNorm *GetAttribute() const { return this->primitive->value_as_BatchNorm(); }
};

class FusedBatchNorm : public Primitive {
 public:
  explicit FusedBatchNorm(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::FusedBatchNorm *GetAttribute() const { return this->primitive->value_as_FusedBatchNorm(); }
};

class Activation : public Primitive {
 public:
  explicit Activation(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Activation *GetAttribute() const { return this->primitive->value_as_Activation(); }
};

class Prelu : public Activation {
 public:
  explicit Prelu(schema::Primitive *primitive) : Activation(primitive) {}
  const schema::Prelu *GetAttribute() const { return this->primitive->value_as_Prelu(); }
};

class CaffePReLU : public Activation {
 public:
  explicit CaffePReLU(schema::Primitive *primitive) : Activation(primitive) {}
  const schema::CaffePReLU *GetAttribute() const { return this->primitive->value_as_CaffePReLU(); }
};

class Split : public Primitive {
 public:
  explicit Split(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Split *GetAttribute() const { return this->primitive->value_as_Split(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Reshape : public Primitive {
 public:
  explicit Reshape(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Reshape *GetAttribute() const { return this->primitive->value_as_Reshape(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;

 private:
  int CalNewShape(const tensor::Tensor *in_tensor, std::vector<int> *out_shape) const;
};

class FullConnection : public Primitive {
 public:
  explicit FullConnection(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::FullConnection *GetAttribute() const { return this->primitive->value_as_FullConnection(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class SoftMax : public Primitive {
 public:
  explicit SoftMax(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::SoftMax *GetAttribute() const { return this->primitive->value_as_SoftMax(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Reduce : public Primitive {
 public:
  explicit Reduce(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Reduce *GetAttribute() const { return this->primitive->value_as_Reduce(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class DepthwiseConv2D : public Primitive {
 public:
  explicit DepthwiseConv2D(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::DepthwiseConv2D *GetAttribute() const { return this->primitive->value_as_DepthwiseConv2D(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
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

class DeConv2D : public Primitive {
 public:
  explicit DeConv2D(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::DeConv2D *GetAttribute() const { return this->primitive->value_as_DeConv2D(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
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

class DeconvDepthwiseConv2D : public Primitive {
 public:
  explicit DeconvDepthwiseConv2D(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::DeDepthwiseConv2D *GetAttribute() const { return this->primitive->value_as_DeDepthwiseConv2D(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
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

class Power : public Primitive {
 public:
  explicit Power(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Power *GetAttribute() const { return this->primitive->value_as_Power(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Range : public Primitive {
 public:
  explicit Range(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Range *GetAttribute() const { return this->primitive->value_as_Range(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class AddN : public Primitive {
 public:
  explicit AddN(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::AddN *GetAttribute() const { return this->primitive->value_as_AddN(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Arithmetic : public Primitive {
 public:
  explicit Arithmetic(schema::Primitive *primitive) : Primitive(primitive) {}
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
  bool Broadcasting() { return this->broadcasting_; }
  int NDims() { return this->ndim_; }
  std::vector<int> InShape0() { return this->in_shape0_; }
  std::vector<int> InShape1() { return this->in_shape1_; }
  std::vector<int> OutputShape() { return this->out_shape_; }

 protected:
  bool broadcasting_ = false;
  int ndim_;
  std::vector<int> in_shape0_;
  std::vector<int> in_shape1_;
  std::vector<int> out_shape_;
};

class Add : public Arithmetic {
 public:
  explicit Add(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Add *GetAttribute() const { return this->primitive->value_as_Add(); }
};

class Mul : public Arithmetic {
 public:
  explicit Mul(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Mul *GetAttribute() const { return this->primitive->value_as_Mul(); }
};

class Sub : public Arithmetic {
 public:
  explicit Sub(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Sub *GetAttribute() const { return this->primitive->value_as_Sub(); }
};

class Div : public Arithmetic {
 public:
  explicit Div(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Div *GetAttribute() const { return this->primitive->value_as_Div(); }
};

class LogicalAnd : public Arithmetic {
 public:
  explicit LogicalAnd(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::LogicalAnd *GetAttribute() const { return this->primitive->value_as_LogicalAnd(); }
};

class LogicalOr : public Arithmetic {
 public:
  explicit LogicalOr(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::LogicalOr *GetAttribute() const { return this->primitive->value_as_LogicalOr(); }
};

class Maximum : public Arithmetic {
 public:
  explicit Maximum(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Maximum *GetAttribute() const { return this->primitive->value_as_Maximum(); }
};

class Minimum : public Arithmetic {
 public:
  explicit Minimum(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Minimum *GetAttribute() const { return this->primitive->value_as_Minimum(); }
};

class FloorDiv : public Arithmetic {
 public:
  explicit FloorDiv(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::FloorDiv *GetAttribute() const { return this->primitive->value_as_FloorDiv(); }
};

class FloorMod : public Arithmetic {
 public:
  explicit FloorMod(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::FloorMod *GetAttribute() const { return this->primitive->value_as_FloorMod(); }
};

class SquaredDifference : public Arithmetic {
 public:
  explicit SquaredDifference(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::SquaredDifference *GetAttribute() const { return this->primitive->value_as_SquaredDifference(); }
};

class Equal : public Arithmetic {
 public:
  explicit Equal(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Equal *GetAttribute() const { return this->primitive->value_as_Equal(); }
};

class NotEqual : public Arithmetic {
 public:
  explicit NotEqual(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::NotEqual *GetAttribute() const { return this->primitive->value_as_NotEqual(); }
};

class Less : public Arithmetic {
 public:
  explicit Less(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Less *GetAttribute() const { return this->primitive->value_as_Less(); }
};

class LessEqual : public Arithmetic {
 public:
  explicit LessEqual(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::LessEqual *GetAttribute() const { return this->primitive->value_as_LessEqual(); }
};

class Greater : public Arithmetic {
 public:
  explicit Greater(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Greater *GetAttribute() const { return this->primitive->value_as_Greater(); }
};

class GreaterEqual : public Arithmetic {
 public:
  explicit GreaterEqual(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::GreaterEqual *GetAttribute() const { return this->primitive->value_as_GreaterEqual(); }
};

class Eltwise : public Arithmetic {
 public:
  explicit Eltwise(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::Eltwise *GetAttribute() const { return this->primitive->value_as_Eltwise(); }
};

class ArithmeticSelf : public Primitive {
 public:
  explicit ArithmeticSelf(schema::Primitive *primitive) : Primitive(primitive) {}
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Abs : public ArithmeticSelf {
 public:
  explicit Abs(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Abs *GetAttribute() const { return this->primitive->value_as_Abs(); }
};

class Cos : public ArithmeticSelf {
 public:
  explicit Cos(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Cos *GetAttribute() const { return this->primitive->value_as_Cos(); }
};

class Exp : public ArithmeticSelf {
 public:
  explicit Exp(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Exp *GetAttribute() const { return this->primitive->value_as_Exp(); }
};

class Log : public ArithmeticSelf {
 public:
  explicit Log(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Log *GetAttribute() const { return this->primitive->value_as_Log(); }
};

class Square : public ArithmeticSelf {
 public:
  explicit Square(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Square *GetAttribute() const { return this->primitive->value_as_Square(); }
};

class Sqrt : public ArithmeticSelf {
 public:
  explicit Sqrt(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Sqrt *GetAttribute() const { return this->primitive->value_as_Sqrt(); }
};

class Rsqrt : public ArithmeticSelf {
 public:
  explicit Rsqrt(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Rsqrt *GetAttribute() const { return this->primitive->value_as_Rsqrt(); }
};

class Sin : public ArithmeticSelf {
 public:
  explicit Sin(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Sin *GetAttribute() const { return this->primitive->value_as_Sin(); }
};

class LogicalNot : public ArithmeticSelf {
 public:
  explicit LogicalNot(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::LogicalNot *GetAttribute() const { return this->primitive->value_as_LogicalNot(); }
};

class Floor : public ArithmeticSelf {
 public:
  explicit Floor(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Floor *GetAttribute() const { return this->primitive->value_as_Floor(); }
};

class Ceil : public ArithmeticSelf {
 public:
  explicit Ceil(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Ceil *GetAttribute() const { return this->primitive->value_as_Ceil(); }
};

class RealDiv : public Arithmetic {
 public:
  explicit RealDiv(schema::Primitive *primitive) : Arithmetic(primitive) {}
  const schema::RealDiv *GetAttribute() const { return this->primitive->value_as_RealDiv(); }
};

class BiasAdd : public Primitive {
 public:
  explicit BiasAdd(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::BiasAdd *GetAttribute() const { return this->primitive->value_as_BiasAdd(); }
};

class ExpandDims : public Primitive {
 public:
  explicit ExpandDims(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ExpandDims *GetAttribute() const { return this->primitive->value_as_ExpandDims(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Unsqueeze : public Primitive {
 public:
  explicit Unsqueeze(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Unsqueeze *GetAttribute() const { return this->primitive->value_as_Unsqueeze(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Cast : public Primitive {
 public:
  explicit Cast(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Cast *GetAttribute() const { return this->primitive->value_as_Cast(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Concat : public Primitive {
 public:
  explicit Concat(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Concat *GetAttribute() const { return this->primitive->value_as_Concat(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Fill : public Primitive {
 public:
  explicit Fill(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Fill *GetAttribute() const { return this->primitive->value_as_Fill(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Mean : public Primitive {
 public:
  explicit Mean(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Mean *GetAttribute() const { return this->primitive->value_as_Mean(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class ArgMax : public Primitive {
 public:
  explicit ArgMax(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ArgMax *GetAttribute() const { return this->primitive->value_as_ArgMax(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class ArgMin : public Primitive {
 public:
  explicit ArgMin(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ArgMin *GetAttribute() const { return this->primitive->value_as_ArgMin(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class MatMul : public Primitive {
 public:
  explicit MatMul(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::MatMul *GetAttribute() const { return this->primitive->value_as_MatMul(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Nchw2Nhwc : public Primitive {
 public:
  explicit Nchw2Nhwc(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Nchw2Nhwc *GetAttribute() const { return this->primitive->value_as_Nchw2Nhwc(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Nhwc2Nchw : public Primitive {
 public:
  explicit Nhwc2Nchw(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Nhwc2Nchw *GetAttribute() const { return this->primitive->value_as_Nhwc2Nchw(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Rank : public Primitive {
 public:
  explicit Rank(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Rank *GetAttribute() const { return this->primitive->value_as_Rank(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Pad : public Primitive {
 public:
  explicit Pad(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Pad *GetAttribute() const { return this->primitive->value_as_Pad(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Gather : public Primitive {
 public:
  explicit Gather(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Gather *GatherAttribute() const { return this->primitive->value_as_Gather(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class GatherNd : public Primitive {
 public:
  explicit GatherNd(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::GatherNd *GetAttribute() const { return this->primitive->value_as_GatherNd(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Slice : public Primitive {
 public:
  explicit Slice(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Slice *GetAttribute() const { return this->primitive->value_as_Slice(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class BroadcastTo : public Primitive {
 public:
  explicit BroadcastTo(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::BroadcastTo *GetAttribute() const { return this->primitive->value_as_BroadcastTo(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Squeeze : public Primitive {
 public:
  explicit Squeeze(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Squeeze *SqueezeAttribute() const { return this->primitive->value_as_Squeeze(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Transpose : public Primitive {
 public:
  explicit Transpose(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Transpose *GetAttribute() const { return this->primitive->value_as_Transpose(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class LocalResponseNormalization : public Primitive {
 public:
  explicit LocalResponseNormalization(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::LocalResponseNormalization *GetAttribute() const {
    return this->primitive->value_as_LocalResponseNormalization();
  }
};

class Tile : public Primitive {
 public:
  explicit Tile(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Tile *GetAttribute() const { return this->primitive->value_as_Tile(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Reverse : public Primitive {
 public:
  explicit Reverse(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Reverse *GetAttribute() const { return this->primitive->value_as_Reverse(); }
};

class TopK : public Primitive {
 public:
  explicit TopK(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::TopK *GetAttribute() const { return this->primitive->value_as_TopK(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};

class Scale : public Primitive {
 public:
  explicit Scale(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Scale *GetAttribute() const { return this->primitive->value_as_Scale(); }
};

class Stack : public Primitive {
 public:
  explicit Stack(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Stack *GetAttribute() const { return this->primitive->value_as_Stack(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Unstack : public Primitive {
 public:
  explicit Unstack(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Unstack *GetAttribute() const { return this->primitive->value_as_Unstack(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Unique : public Primitive {
 public:
  explicit Unique(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Unique *GetAttribute() const { return this->primitive->value_as_Unique(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class ReverseSequence : public Primitive {
 public:
  explicit ReverseSequence(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ReverseSequence *GetAttribute() const { return this->primitive->value_as_ReverseSequence(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class DepthToSpace : public Primitive {
 public:
  explicit DepthToSpace(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::DepthToSpace *GetAttribute() const { return this->primitive->value_as_DepthToSpace(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Resize : public Primitive {
 public:
  explicit Resize(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Resize *GetAttrbute() const { return this->primitive->value_as_Resize(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Round : public ArithmeticSelf {
 public:
  explicit Round(schema::Primitive *primitive) : ArithmeticSelf(primitive) {}
  const schema::Round *GetAttribute() const { return this->primitive->value_as_Round(); }
};

class ZerosLike : public Primitive {
 public:
  explicit ZerosLike(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ZerosLike *GetAttribute() const { return this->primitive->value_as_ZerosLike(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Where : public Primitive {
 public:
  explicit Where(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Where *GetAttribute() const { return this->primitive->value_as_Where(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class BatchToSpace : public Primitive {
 public:
  explicit BatchToSpace(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::BatchToSpace *GetAttribute() const { return this->primitive->value_as_BatchToSpace(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class SpaceToBatch : public Primitive {
 public:
  explicit SpaceToBatch(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::SpaceToBatch *GetAttribute() const { return this->primitive->value_as_SpaceToBatch(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
  std::vector<int> BlockSizes() { return block_sizes_; }
  std::vector<int> Paddings() { return block_sizes_; }
  std::vector<int> InShape() { return block_sizes_; }
  std::vector<int> PaddedInShape() { return block_sizes_; }

 private:
  std::vector<int> block_sizes_;
  std::vector<int> paddings_;
  std::vector<int> in_shape_;
  std::vector<int> padded_in_shape_;
};

class Crop : public Primitive {
 public:
  explicit Crop(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Crop *GetAttribute() const { return this->primitive->value_as_Crop(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Shape : public Primitive {
 public:
  explicit Shape(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Shape *GetAttribute() const { return this->primitive->value_as_Shape(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class ScatterND : public Primitive {
 public:
  explicit ScatterND(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::ScatterND *GetAttribute() const { return this->primitive->value_as_ScatterND(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Flatten : public Primitive {
 public:
  explicit Flatten(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Flatten *GetAttribute() const { return this->primitive->value_as_Flatten(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class OneHot : public Primitive {
 public:
  explicit OneHot(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::OneHot *GetAttribute() const { return this->primitive->value_as_OneHot(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class StridedSlice : public Primitive {
 public:
  explicit StridedSlice(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::StridedSlice *GetAttribute() const { return this->primitive->value_as_StridedSlice(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
  int NDims() { return this->ndim_; }
  void ApplyNewAxisMask();
  std::vector<int> ApplyShrinkMask(std::vector<int> out_shape);
  void ApplyBeginMask();
  void ApplyEndMask();
  void ApplyEllipsisMask();
  std::vector<int> GetInShape() { return this->in_shape_; }
  std::vector<int> GetBegins() { return this->begins_; }
  std::vector<int> GetEnds() { return this->ends_; }
  std::vector<int> GetStrides() { return this->strides_; }

 protected:
  int ndim_;
  std::vector<int> in_shape_;
  std::vector<int> begins_;
  std::vector<int> ends_;
  std::vector<int> strides_;
  std::vector<bool> begins_mask_;
  std::vector<bool> ends_mask_;
  std::vector<bool> ellipsis_mask_;
  std::vector<bool> new_axis_mask_;
  std::vector<bool> shrink_axis_mask_;
};

class PriorBox : public Primitive {
 public:
  explicit PriorBox(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::PriorBox *GetAttrbute() const { return this->primitive->value_as_PriorBox(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class SpaceToDepth : public Primitive {
 public:
  explicit SpaceToDepth(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::SpaceToDepth *GetAttribute() const { return this->primitive->value_as_SpaceToDepth(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class QuantDTypeCast : public Primitive {
 public:
  explicit QuantDTypeCast(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::QuantDTypeCast *GetAttribute() const { return this->primitive->value_as_QuantDTypeCast(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Lstm : public Primitive {
 public:
  explicit Lstm(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Lstm *GetAttribute() const { return this->primitive->value_as_Lstm(); }
  int InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) override;
};

class Elu : public Primitive {
 public:
  explicit Elu(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::Elu *GetAttribute() const { return this->primitive->value_as_Elu(); }
};

class EmbeddingLookup : public Primitive {
 public:
  explicit EmbeddingLookup(schema::Primitive *primitive) : Primitive(primitive) {}
  const schema::EmbeddingLookup *GetAttribute() const { return this->primitive->value_as_EmbeddingLookup(); }
  int InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_OPS_OPS_H_
