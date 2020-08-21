/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/primitive_c.h"
#include <memory>
#include "src/ops/space_to_batch.h"
#include "src/ops/conv2d.h"
#include "src/ops/roi_pooling.h"
#include "src/ops/topk.h"
#include "src/ops/broadcast_to.h"
#include "src/ops/unsqueeze.h"
#include "src/ops/unstack.h"
#include "src/ops/depth_to_space.h"
#include "src/ops/batch_to_space.h"
#include "src/ops/prior_box.h"
#include "src/ops/lstm.h"
#include "src/ops/softmax.h"
#include "src/ops/activation.h"
#include "src/ops/deconv2d.h"
#include "src/ops/reduce.h"
#include "src/ops/pooling.h"
#include "src/ops/fused_batchnorm.h"
#include "src/ops/batch_norm.h"
#include "src/ops/power.h"
#include "src/ops/range.h"
#include "src/ops/add.h"
#include "src/ops/sub.h"
#include "src/ops/div.h"
#include "src/ops/bias_add.h"
#include "src/ops/expand_dims.h"
#include "src/ops/full_connection.h"
#include "src/ops/shape.h"
#include "src/ops/elu.h"
#include "src/ops/embedding_lookup.h"
#include "src/ops/quant_dtype_cast.h"
#include "src/ops/matmul.h"
#include "src/ops/resize.h"
#include "src/ops/tile.h"
#include "src/ops/one_hot.h"
#include "src/ops/space_to_depth.h"
#include "src/ops/split.h"
#include "src/ops/argmax.h"
#include "src/ops/argmin.h"
#include "src/ops/cast.h"
#include "src/ops/reshape.h"
#include "src/ops/scale.h"
#include "src/ops/concat.h"
#include "src/ops/nchw2nhwc.h"
#include "src/ops/slice.h"
#include "src/ops/squeeze.h"
#include "src/ops/flatten.h"
#include "src/ops/mean.h"
#include "src/ops/nhwc2nchw.h"
#include "src/ops/stack.h"
#include "src/ops/crop.h"
#include "src/ops/addn.h"
#include "src/ops/gather.h"
#include "src/ops/gather_nd.h"
#include "src/ops/local_response_normalization.h"
#include "src/ops/pad.h"
#include "src/ops/prelu.h"
#include "src/ops/caffe_p_relu.h"
#include "src/ops/reverse_sequence.h"
#include "src/ops/dedepthwise_conv2d.h"
#include "src/ops/depthwise_conv2d.h"
#include "src/ops/mul.h"
#include "src/ops/eltwise.h"
#include "src/ops/fill.h"
#include "src/ops/transpose.h"
#include "src/ops/log.h"
#include "src/ops/abs.h"
#include "src/ops/sin.h"
#include "src/ops/cos.h"
#include "src/ops/sqrt.h"
#include "src/ops/square.h"
#include "src/ops/exp.h"
#include "src/ops/rsqrt.h"
#include "src/ops/maximum.h"
#include "src/ops/minimum.h"
#include "src/ops/strided_slice.h"
#include "src/ops/reverse.h"
#include "src/ops/logical_and.h"
#include "src/ops/logical_or.h"
#include "src/ops/logical_not.h"
#include "src/ops/floor_div.h"
#include "src/ops/floor_mod.h"
#include "src/ops/equal.h"
#include "src/ops/not_equal.h"
#include "src/ops/less.h"
#include "src/ops/less_equal.h"
#include "src/ops/greater_equal.h"
#include "src/ops/greater.h"
#include "src/ops/floor.h"
#include "src/ops/squared_difference.h"
#include "src/ops/ceil.h"
#include "src/ops/round.h"
#include "src/ops/unique.h"
#include "src/ops/zeros_like.h"
#include "src/ops/where.h"
#include "src/ops/scatter_nd.h"
#include "src/ops/constant_of_shape.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
schema::PrimitiveT *PrimitiveC::GetPrimitiveT() const { return this->primitive_; }

void PrimitiveC::SetPrimitiveT(schema::PrimitiveT *prim) { this->primitive_ = prim; }

void PrimitiveC::SetInputQuantParam(const std::vector<std::vector<schema::QuantParamT>> &input_quant_param) {
  this->input_quant_param_ = input_quant_param;
}

void PrimitiveC::SetOutputQuantParam(const std::vector<std::vector<schema::QuantParamT>> &output_quant_param) {
  this->output_quant_param_ = output_quant_param;
}

void PrimitiveC::ClearInputOutputQuantParam() {
  input_quant_param_.clear();
  output_quant_param_.clear();
}

void PrimitiveC::AddInputQuantParam(std::vector<schema::QuantParamT> quant_param) {
  this->input_quant_param_.emplace_back(quant_param);
}
std::vector<std::vector<schema::QuantParamT>> PrimitiveC::GetInputQuantParams() const { return input_quant_param_; }

void PrimitiveC::AddOutputQuantParam(std::vector<schema::QuantParamT> quant_param) {
  this->output_quant_param_.emplace_back(quant_param);
}
std::vector<std::vector<schema::QuantParamT>> PrimitiveC::GetOutputQuantParams() const { return output_quant_param_; }

void PrimitiveC::SetQuantType(schema::QuantType quant_type) { this->quant_type_ = quant_type; }

schema::QuantType PrimitiveC::GetQuantType() const { return quant_type_; }

std::shared_ptr<PrimitiveC> GetReturnPrim() {
  auto return_primitiveT = new schema::PrimitiveT;
  return_primitiveT->value.type = schema::PrimitiveType_Return;
  return_primitiveT->value.value = new schema::ReturnT;
  return std::make_shared<PrimitiveC>(return_primitiveT);
}

std::shared_ptr<PrimitiveC> GetMakeTuplePrim() {
  auto make_tuple_primitiveT = new schema::PrimitiveT;
  make_tuple_primitiveT->value.type = schema::PrimitiveType_MakeTuple;
  make_tuple_primitiveT->value.value = new schema::MakeTupleT;
  return std::make_shared<PrimitiveC>(make_tuple_primitiveT);
}

std::shared_ptr<PrimitiveC> GetTupleGetItemPrim() {
  auto tuple_get_item_primitiveT = new schema::PrimitiveT();
  tuple_get_item_primitiveT->value.type = schema::PrimitiveType_TupleGetItem;
  tuple_get_item_primitiveT->value.value = new schema::TupleGetItemT;
  return std::make_shared<PrimitiveC>(tuple_get_item_primitiveT);
}

PrimitiveC *PrimitiveC::CreatePrimitive(mindspore::schema::Primitive *primitive) {
  MS_ASSERT(primitive != nullptr);
  auto op_type = primitive->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new SoftMax(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Activation:
      return new Activation(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Conv2D:
      return new Conv2D(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reduce:
      return new Reduce(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Pooling:
      return new Pooling(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ROIPooling:
      return new ROIPooling(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_DepthwiseConv2D:
      return new DepthwiseConv2D(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FusedBatchNorm:
      return new FusedBatchNorm(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_BatchNorm:
      return new BatchNorm(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FullConnection:
      return new FullConnection(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Power:
      return new Power(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Pad:
      return new Pad(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Range:
      return new Range(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Mul:
      return new Mul(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Add:
      return new Add(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Sub:
      return new Sub(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Div:
      return new Div(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_BiasAdd:
      return new BiasAdd(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ExpandDims:
      return new ExpandDims(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ArgMax:
      return new ArgMax(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ArgMin:
      return new ArgMin(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Cast:
      return new Cast(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reshape:
      return new Reshape(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Eltwise:
      return new Eltwise(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Ceil:
      return new Ceil(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Concat:
      return new Concat(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Fill:
      return new Fill(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Nhwc2Nchw:
      return new Nhwc2Nchw(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Nchw2Nhwc:
      return new Nchw2Nhwc(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Transpose:
      return new Transpose(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Squeeze:
      return new Squeeze(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SquaredDifference:
      return new SquaredDifference(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Split:
      return new Split(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FloorDiv:
      return new FloorDiv(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FloorMod:
      return new FloorMod(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reverse:
      return new Reverse(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Scale:
      return new Scale(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_GatherNd:
      return new GatherNd(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Tile:
      return new Tile(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_TopK:
      return new TopK(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unique:
      return new Unique(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unstack:
      return new Unstack(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ReverseSequence:
      return new ReverseSequence(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Round:
      return new Round(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ZerosLike:
      return new ZerosLike(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Where:
      return new Where(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Floor:
      return new Floor(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Shape:
      return new Shape(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ScatterND:
      return new ScatterND(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unsqueeze:
      return new Unsqueeze(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Flatten:
      return new Flatten(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_StridedSlice:
      return new StridedSlice(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Resize:
      return new Resize(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_OneHot:
      return new OneHot(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_PriorBox:
      return new PriorBox(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SpaceToDepth:
      return new SpaceToDepth(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SpaceToBatch:
      return new SpaceToBatch(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_QuantDTypeCast:
      return new QuantDTypeCast(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_MatMul:
      return new MatMul(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_EmbeddingLookup:
      return new EmbeddingLookup(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ConstantOfShape:
      return new ConstantOfShape(const_cast<schema::Primitive *>(primitive));
    default:
      break;
  }
  return nullptr;
}

#endif

int PrimitiveC::Type() const {
#ifdef PRIMITIVE_WRITEABLE
  return this->primitive_->value.type;
#else
  return this->primitive_->value_type();
#endif
}
bool PrimitiveC::GetInferFlag() const { return this->infer_flag_; }

void PrimitiveC::SetInferFlag(bool flag) { this->infer_flag_ = flag; }

int PrimitiveC::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  return 0;
}

}  // namespace lite
}  // namespace mindspore
