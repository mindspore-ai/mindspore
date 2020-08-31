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
#include "src/ops/space_to_batch_nd.h"
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
#include "src/ops/p_relu.h"
#include "src/ops/leaky_relu.h"
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
#include "src/ops/return.h"
#include "src/ops/where.h"
#include "src/ops/scatter_nd.h"
#include "src/ops/constant_of_shape.h"
#include "src/ops/dequant.h"
#include "src/ops/make_tuple.h"
#include "src/ops/quant.h"
#include "src/ops/tuple_get_item.h"
#include "src/ops/l2_norm.h"
#include "src/ops/sparse_to_dense.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
schema::PrimitiveT *PrimitiveC::GetPrimitiveT() const { return this->primitive_; }

void PrimitiveC::ClearPrimitiveT() { this->primitive_ = nullptr; }

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
  auto return_primitiveT = new (std::nothrow) schema::PrimitiveT;
  if (return_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  return_primitiveT->value.type = schema::PrimitiveType_Return;
  return_primitiveT->value.value = new schema::ReturnT;
  if (return_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new ReturnT failed";
    delete (return_primitiveT);
    return nullptr;
  }
  return std::make_shared<Return>(return_primitiveT);
}

std::shared_ptr<PrimitiveC> GetMakeTuplePrim() {
  auto make_tuple_primitiveT = new schema::PrimitiveT;
  if (make_tuple_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  make_tuple_primitiveT->value.type = schema::PrimitiveType_MakeTuple;
  make_tuple_primitiveT->value.value = new schema::MakeTupleT;
  if (make_tuple_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new MakeTupleT failed";
    delete (make_tuple_primitiveT);
    return nullptr;
  }
  return std::make_shared<MakeTuple>(make_tuple_primitiveT);
}

std::shared_ptr<PrimitiveC> GetTupleGetItemPrim() {
  auto tuple_get_item_primitiveT = new schema::PrimitiveT();
  if (tuple_get_item_primitiveT == nullptr) {
    MS_LOG(ERROR) << "new PrimitiveT failed";
    return nullptr;
  }
  tuple_get_item_primitiveT->value.type = schema::PrimitiveType_TupleGetItem;
  tuple_get_item_primitiveT->value.value = new schema::TupleGetItemT;
  if (tuple_get_item_primitiveT->value.value == nullptr) {
    MS_LOG(ERROR) << "new TupleGetItemT failed";
    delete (tuple_get_item_primitiveT);
    return nullptr;
  }
  return std::make_shared<TupleGetItem>(tuple_get_item_primitiveT);
}

template <typename T, typename = std::enable_if<std::is_base_of<PrimitiveC, T>::value>>
std::shared_ptr<PrimitiveC> NewPrimitiveC(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  auto primc = std::make_shared<T>();
  if (primc == nullptr) {
    MS_LOG(ERROR) << "make_shared PrimitiveC failed";
    return nullptr;
  }
  auto ret = primc->UnPackAttr(prim, inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnPackAttr failed";
    return nullptr;
  }
  return primc;
}

std::shared_ptr<PrimitiveC> PrimitiveC::UnPackFromPrimitive(const Primitive &prim,
                                                            const std::vector<AnfNodePtr> &inputs) {
  const auto &op_type = prim.name();
  if (op_type == "ReLU" || op_type == "ReLU6" || op_type == "Sigmoid") {
    return NewPrimitiveC<Activation>(prim, inputs);
  } else if (op_type == "BatchNorm") {
    return NewPrimitiveC<BatchNorm>(prim, inputs);
  } else if (op_type == "BiasAdd") {
    return NewPrimitiveC<BiasAdd>(prim, inputs);
  } else if (op_type == "Concat") {
    return NewPrimitiveC<Concat>(prim, inputs);
  } else if (op_type == "Conv2D") {
    return NewPrimitiveC<Conv2D>(prim, inputs);
  } else if (op_type == "DepthwiseConv2dNative" || op_type == "DepthwiseConv2D") {
    return NewPrimitiveC<DepthwiseConv2D>(prim, inputs);
  } else if (op_type == "Dequant") {
    return NewPrimitiveC<Dequant>(prim, inputs);
  } else if (op_type == "Flatten") {
    return NewPrimitiveC<Flatten>(prim, inputs);
  } else if (op_type == "make_tuple") {
    return NewPrimitiveC<MakeTuple>(prim, inputs);
  } else if (op_type == "MatMul") {
    return NewPrimitiveC<MatMul>(prim, inputs);
  } else if (op_type == "Mul") {
    return NewPrimitiveC<Mul>(prim, inputs);
  } else if (op_type == "MaxPool") {
    return NewPrimitiveC<Pooling>(prim, inputs);
  } else if (op_type == "Quant") {
    return NewPrimitiveC<Quant>(prim, inputs);
  } else if (op_type == "ReduceMean") {
    return NewPrimitiveC<Reduce>(prim, inputs);
  } else if (op_type == "Reshape") {
    return NewPrimitiveC<Reshape>(prim, inputs);
  } else if (op_type == "TensorAdd") {
    return NewPrimitiveC<Add>(prim, inputs);
  } else if (op_type == "Transpose") {
    return NewPrimitiveC<Transpose>(prim, inputs);
  } else if (op_type == "tuple_getitem") {
    return NewPrimitiveC<TupleGetItem>(prim, inputs);
  } else if (op_type == "Softmax") {
    return NewPrimitiveC<SoftMax>(prim, inputs);
  } else {
    MS_LOG(ERROR) << "Unsupported primitive type in UnPackFromPrimitive : " << op_type;
    return nullptr;
  }
}

PrimitiveC *PrimitiveC::UnPackFromSchemaPrimitiveT(mindspore::schema::PrimitiveT *primitive) {
  MS_ASSERT(primitive != nullptr);
  auto op_type = primitive->value.type;
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new SoftMax(primitive);
    case schema::PrimitiveType_Activation:
      return new Activation(primitive);
    case schema::PrimitiveType_Conv2D:
      return new Conv2D(primitive);
    case schema::PrimitiveType_DeConv2D:
      return new DeConv2D(primitive);
    case schema::PrimitiveType_Reduce:
      return new Reduce(primitive);
    case schema::PrimitiveType_Pooling:
      return new Pooling(primitive);
    case schema::PrimitiveType_ROIPooling:
      return new ROIPooling(primitive);
    case schema::PrimitiveType_DepthwiseConv2D:
      return new DepthwiseConv2D(primitive);
    case schema::PrimitiveType_FusedBatchNorm:
      return new FusedBatchNorm(primitive);
    case schema::PrimitiveType_BatchNorm:
      return new BatchNorm(primitive);
    case schema::PrimitiveType_FullConnection:
      return new FullConnection(primitive);
    case schema::PrimitiveType_Power:
      return new Power(primitive);
    case schema::PrimitiveType_Pad:
      return new Pad(primitive);
    case schema::PrimitiveType_Range:
      return new Range(primitive);
    case schema::PrimitiveType_Mul:
      return new Mul(primitive);
    case schema::PrimitiveType_Add:
      return new Add(primitive);
    case schema::PrimitiveType_Sub:
      return new Sub(primitive);
    case schema::PrimitiveType_Div:
      return new Div(primitive);
    case schema::PrimitiveType_BiasAdd:
      return new BiasAdd(primitive);
    case schema::PrimitiveType_ExpandDims:
      return new ExpandDims(primitive);
    case schema::PrimitiveType_ArgMax:
      return new ArgMax(primitive);
    case schema::PrimitiveType_ArgMin:
      return new ArgMin(primitive);
    case schema::PrimitiveType_Cast:
      return new Cast(primitive);
    case schema::PrimitiveType_Reshape:
      return new Reshape(primitive);
    case schema::PrimitiveType_Scale:
      return new Scale(primitive);
    case schema::PrimitiveType_Eltwise:
      return new Eltwise(primitive);
    case schema::PrimitiveType_Ceil:
      return new Ceil(primitive);
    case schema::PrimitiveType_Concat:
      return new Concat(primitive);
    case schema::PrimitiveType_Fill:
      return new Fill(primitive);
    case schema::PrimitiveType_Nhwc2Nchw:
      return new Nhwc2Nchw(primitive);
    case schema::PrimitiveType_Nchw2Nhwc:
      return new Nchw2Nhwc(primitive);
    case schema::PrimitiveType_Transpose:
      return new Transpose(primitive);
    case schema::PrimitiveType_Slice:
      return new Slice(primitive);
    case schema::PrimitiveType_Squeeze:
      return new Squeeze(primitive);
    case schema::PrimitiveType_Flatten:
      return new Flatten(primitive);
    case schema::PrimitiveType_Mean:
      return new Mean(primitive);
    case schema::PrimitiveType_Stack:
      return new Stack(primitive);
    case schema::PrimitiveType_Crop:
      return new Crop(primitive);
    case schema::PrimitiveType_SquaredDifference:
      return new SquaredDifference(primitive);
    case schema::PrimitiveType_AddN:
      return new AddN(primitive);
    case schema::PrimitiveType_Abs:
      return new Abs(primitive);
    case schema::PrimitiveType_Sin:
      return new Sin(primitive);
    case schema::PrimitiveType_Cos:
      return new Cos(primitive);
    case schema::PrimitiveType_Log:
      return new Log(primitive);
    case schema::PrimitiveType_Sqrt:
      return new Sqrt(primitive);
    case schema::PrimitiveType_Rsqrt:
      return new Rsqrt(primitive);
    case schema::PrimitiveType_Square:
      return new Square(primitive);
    case schema::PrimitiveType_Exp:
      return new Exp(primitive);
    case schema::PrimitiveType_Gather:
      return new Gather(primitive);
    case schema::PrimitiveType_GatherNd:
      return new GatherNd(primitive);
    case schema::PrimitiveType_LocalResponseNormalization:
      return new LocalResponseNormalization(primitive);
    case schema::PrimitiveType_Maximum:
      return new Maximum(primitive);
    case schema::PrimitiveType_Minimum:
      return new Minimum(primitive);
    case schema::PrimitiveType_StridedSlice:
      return new StridedSlice(primitive);
    case schema::PrimitiveType_LeakyReLU:
      return new (std::nothrow) LeakyReLU(primitive);
    case schema::PrimitiveType_PReLU:
      return new (std::nothrow) PReLU(primitive);
    case schema::PrimitiveType_Round:
      return new Round(primitive);
    case schema::PrimitiveType_Reverse:
      return new Reverse(primitive);
    case schema::PrimitiveType_ReverseSequence:
      return new ReverseSequence(primitive);
    case schema::PrimitiveType_LogicalAnd:
      return new LogicalAnd(primitive);
    case schema::PrimitiveType_LogicalOr:
      return new LogicalOr(primitive);
    case schema::PrimitiveType_LogicalNot:
      return new LogicalNot(primitive);
    case schema::PrimitiveType_FloorDiv:
      return new FloorDiv(primitive);
    case schema::PrimitiveType_FloorMod:
      return new FloorMod(primitive);
    case schema::PrimitiveType_Equal:
      return new Equal(primitive);
    case schema::PrimitiveType_NotEqual:
      return new NotEqual(primitive);
    case schema::PrimitiveType_Less:
      return new Less(primitive);
    case schema::PrimitiveType_LessEqual:
      return new LessEqual(primitive);
    case schema::PrimitiveType_Greater:
      return new Greater(primitive);
    case schema::PrimitiveType_GreaterEqual:
      return new GreaterEqual(primitive);
    case schema::PrimitiveType_Floor:
      return new Floor(primitive);
    case schema::PrimitiveType_Split:
      return new Split(primitive);
    case schema::PrimitiveType_OneHot:
      return new OneHot(primitive);
    case schema::PrimitiveType_PriorBox:
      return new PriorBox(primitive);
    case schema::PrimitiveType_SpaceToDepth:
      return new SpaceToDepth(primitive);
    case schema::PrimitiveType_Tile:
      return new Tile(primitive);
    case schema::PrimitiveType_Resize:
      return new Resize(primitive);
    case schema::PrimitiveType_Unstack:
      return new Unstack(primitive);
    case schema::PrimitiveType_Unique:
      return new Unique(primitive);
    case schema::PrimitiveType_TopK:
      return new TopK(primitive);
    case schema::PrimitiveType_MatMul:
      return new MatMul(primitive);
    case schema::PrimitiveType_QuantDTypeCast:
      return new QuantDTypeCast(primitive);
    case schema::PrimitiveType_EmbeddingLookup:
      return new EmbeddingLookup(primitive);
    case schema::PrimitiveType_Elu:
      return new Elu(primitive);
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return new DeDepthwiseConv2D(primitive);
    case schema::PrimitiveType_Shape:
      return new Shape(primitive);
    case schema::PrimitiveType_Unsqueeze:
      return new Unsqueeze(primitive);
    case schema::PrimitiveType_BatchToSpace:
      return new BatchToSpace(primitive);
    case schema::PrimitiveType_SpaceToBatch:
      return new SpaceToBatch(primitive);
    case schema::PrimitiveType_SpaceToBatchND:
      return new SpaceToBatchND(primitive);
    case schema::PrimitiveType_BroadcastTo:
      return new BroadcastTo(primitive);
    case schema::PrimitiveType_DepthToSpace:
      return new DepthToSpace(primitive);
    case schema::PrimitiveType_Lstm:
      return new Lstm(primitive);
    case schema::PrimitiveType_ZerosLike:
      return new ZerosLike(primitive);
    case schema::PrimitiveType_MakeTuple:
      return new MakeTuple(primitive);
    case schema::PrimitiveType_Where:
      return new Where(primitive);
    case schema::PrimitiveType_ScatterND:
      return new ScatterND(primitive);
    case schema::PrimitiveType_ConstantOfShape:
      return new ConstantOfShape(primitive);
    case schema::PrimitiveType_L2Norm:
      return new L2Norm(primitive);
    case schema::PrimitiveType_SparseToDense:
      return new SparseToDense(primitive);
    default:
      MS_LOG(ERROR) << "Unsupported primitive type in UnPackFromSchemaPrimitiveT : "
                    << schema::EnumNamePrimitiveType(op_type);
      break;
  }
  return nullptr;
}
#else
PrimitiveC *PrimitiveC::UnPackFromSchemaPrimitive(const schema::Primitive *primitive) {
  MS_ASSERT(primitive);
  auto op_type = primitive->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return NewPrimitiveC<SoftMax>(primitive);
    case schema::PrimitiveType_Activation:
      return NewPrimitiveC<Activation>(primitive);
    case schema::PrimitiveType_Conv2D:
      return NewPrimitiveC<Conv2D>(primitive);
    case schema::PrimitiveType_DeConv2D:
      return NewPrimitiveC<DeConv2D>(primitive);
    case schema::PrimitiveType_Reduce:
      return NewPrimitiveC<Reduce>(primitive);
    case schema::PrimitiveType_Pooling:
      return NewPrimitiveC<Pooling>(primitive);
    case schema::PrimitiveType_ROIPooling:
      return NewPrimitiveC<ROIPooling>(primitive);
    case schema::PrimitiveType_DepthwiseConv2D:
      return NewPrimitiveC<DepthwiseConv2D>(primitive);
    case schema::PrimitiveType_FusedBatchNorm:
      return NewPrimitiveC<FusedBatchNorm>(primitive);
    case schema::PrimitiveType_BatchNorm:
      return NewPrimitiveC<BatchNorm>(primitive);
    case schema::PrimitiveType_FullConnection:
      return NewPrimitiveC<FullConnection>(primitive);
    case schema::PrimitiveType_Power:
      return NewPrimitiveC<Power>(primitive);
    case schema::PrimitiveType_Pad:
      return NewPrimitiveC<Pad>(primitive);
    case schema::PrimitiveType_Range:
      return NewPrimitiveC<Range>(primitive);
    case schema::PrimitiveType_Mul:
      return NewPrimitiveC<Mul>(primitive);
    case schema::PrimitiveType_Add:
      return NewPrimitiveC<Add>(primitive);
    case schema::PrimitiveType_Sub:
      return NewPrimitiveC<Sub>(primitive);
    case schema::PrimitiveType_Div:
      return NewPrimitiveC<Div>(primitive);
    case schema::PrimitiveType_BiasAdd:
      return NewPrimitiveC<BiasAdd>(primitive);
    case schema::PrimitiveType_ExpandDims:
      return NewPrimitiveC<ExpandDims>(primitive);
    case schema::PrimitiveType_ArgMax:
      return NewPrimitiveC<ArgMax>(primitive);
    case schema::PrimitiveType_ArgMin:
      return NewPrimitiveC<ArgMin>(primitive);
    case schema::PrimitiveType_Cast:
      return NewPrimitiveC<Cast>(primitive);
    case schema::PrimitiveType_Reshape:
      return NewPrimitiveC<Reshape>(primitive);
    case schema::PrimitiveType_Scale:
      return NewPrimitiveC<Scale>(primitive);
    case schema::PrimitiveType_Eltwise:
      return NewPrimitiveC<Eltwise>(primitive);
    case schema::PrimitiveType_Ceil:
      return NewPrimitiveC<Ceil>(primitive);
    case schema::PrimitiveType_Concat:
      return NewPrimitiveC<Concat>(primitive);
    case schema::PrimitiveType_Fill:
      return NewPrimitiveC<Fill>(primitive);
    case schema::PrimitiveType_Nhwc2Nchw:
      return NewPrimitiveC<Nhwc2Nchw>(primitive);
    case schema::PrimitiveType_Nchw2Nhwc:
      return NewPrimitiveC<Nchw2Nhwc>(primitive);
    case schema::PrimitiveType_Transpose:
      return NewPrimitiveC<Transpose>(primitive);
    case schema::PrimitiveType_Slice:
      return NewPrimitiveC<Slice>(primitive);
    case schema::PrimitiveType_Squeeze:
      return NewPrimitiveC<Squeeze>(primitive);
    case schema::PrimitiveType_Flatten:
      return NewPrimitiveC<Flatten>(primitive);
    case schema::PrimitiveType_Mean:
      return NewPrimitiveC<Mean>(primitive);
    case schema::PrimitiveType_Stack:
      return NewPrimitiveC<Stack>(primitive);
    case schema::PrimitiveType_Crop:
      return NewPrimitiveC<Crop>(primitive);
    case schema::PrimitiveType_SquaredDifference:
      return NewPrimitiveC<SquaredDifference>(primitive);
    case schema::PrimitiveType_AddN:
      return NewPrimitiveC<AddN>(primitive);
    case schema::PrimitiveType_Abs:
      return NewPrimitiveC<Abs>(primitive);
    case schema::PrimitiveType_Sin:
      return NewPrimitiveC<Sin>(primitive);
    case schema::PrimitiveType_Cos:
      return NewPrimitiveC<Cos>(primitive);
    case schema::PrimitiveType_Log:
      return NewPrimitiveC<Log>(primitive);
    case schema::PrimitiveType_Sqrt:
      return NewPrimitiveC<Sqrt>(primitive);
    case schema::PrimitiveType_Rsqrt:
      return NewPrimitiveC<Rsqrt>(primitive);
    case schema::PrimitiveType_Square:
      return NewPrimitiveC<Square>(primitive);
    case schema::PrimitiveType_Exp:
      return NewPrimitiveC<Exp>(primitive);
    case schema::PrimitiveType_Gather:
      return NewPrimitiveC<Gather>(primitive);
    case schema::PrimitiveType_GatherNd:
      return NewPrimitiveC<GatherNd>(primitive);
    case schema::PrimitiveType_LocalResponseNormalization:
      return NewPrimitiveC<LocalResponseNormalization>(primitive);
    case schema::PrimitiveType_Maximum:
      return NewPrimitiveC<Maximum>(primitive);
    case schema::PrimitiveType_Minimum:
      return NewPrimitiveC<Minimum>(primitive);
    case schema::PrimitiveType_StridedSlice:
      return NewPrimitiveC<StridedSlice>(primitive);
    case schema::PrimitiveType_LeakyReLU:
      return NewPrimitiveC<LeakyReLU>(primitive);
    case schema::PrimitiveType_PReLU:
      return NewPrimitiveC<PReLU>(primitive);
    case schema::PrimitiveType_Round:
      return NewPrimitiveC<Round>(primitive);
    case schema::PrimitiveType_Reverse:
      return NewPrimitiveC<Reverse>(primitive);
    case schema::PrimitiveType_ReverseSequence:
      return NewPrimitiveC<ReverseSequence>(primitive);
    case schema::PrimitiveType_LogicalAnd:
      return NewPrimitiveC<LogicalAnd>(primitive);
    case schema::PrimitiveType_LogicalOr:
      return NewPrimitiveC<LogicalOr>(primitive);
    case schema::PrimitiveType_LogicalNot:
      return NewPrimitiveC<LogicalNot>(primitive);
    case schema::PrimitiveType_FloorDiv:
      return NewPrimitiveC<FloorDiv>(primitive);
    case schema::PrimitiveType_FloorMod:
      return NewPrimitiveC<FloorMod>(primitive);
    case schema::PrimitiveType_Equal:
      return NewPrimitiveC<Equal>(primitive);
    case schema::PrimitiveType_NotEqual:
      return NewPrimitiveC<NotEqual>(primitive);
    case schema::PrimitiveType_Less:
      return NewPrimitiveC<Less>(primitive);
    case schema::PrimitiveType_LessEqual:
      return NewPrimitiveC<LessEqual>(primitive);
    case schema::PrimitiveType_Greater:
      return NewPrimitiveC<Greater>(primitive);
    case schema::PrimitiveType_GreaterEqual:
      return NewPrimitiveC<GreaterEqual>(primitive);
    case schema::PrimitiveType_Floor:
      return NewPrimitiveC<Floor>(primitive);
    case schema::PrimitiveType_Split:
      return NewPrimitiveC<Split>(primitive);
    case schema::PrimitiveType_OneHot:
      return NewPrimitiveC<OneHot>(primitive);
    case schema::PrimitiveType_PriorBox:
      return NewPrimitiveC<PriorBox>(primitive);
    case schema::PrimitiveType_SpaceToDepth:
      return NewPrimitiveC<SpaceToDepth>(primitive);
    case schema::PrimitiveType_Tile:
      return NewPrimitiveC<Tile>(primitive);
    case schema::PrimitiveType_Resize:
      return NewPrimitiveC<Resize>(primitive);
    case schema::PrimitiveType_Unstack:
      return NewPrimitiveC<Unstack>(primitive);
    case schema::PrimitiveType_Unique:
      return NewPrimitiveC<Unique>(primitive);
    case schema::PrimitiveType_TopK:
      return NewPrimitiveC<TopK>(primitive);
    case schema::PrimitiveType_MatMul:
      return NewPrimitiveC<MatMul>(primitive);
    case schema::PrimitiveType_QuantDTypeCast:
      return NewPrimitiveC<QuantDTypeCast>(primitive);
    case schema::PrimitiveType_EmbeddingLookup:
      return NewPrimitiveC<EmbeddingLookup>(primitive);
    case schema::PrimitiveType_Elu:
      return NewPrimitiveC<Elu>(primitive);
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return NewPrimitiveC<DeDepthwiseConv2D>(primitive);
    case schema::PrimitiveType_Shape:
      return NewPrimitiveC<Shape>(primitive);
    case schema::PrimitiveType_Unsqueeze:
      return NewPrimitiveC<Unsqueeze>(primitive);
    case schema::PrimitiveType_BatchToSpace:
      return NewPrimitiveC<BatchToSpace>(primitive);
    case schema::PrimitiveType_SpaceToBatch:
      return NewPrimitiveC<SpaceToBatch>(primitive);
    case schema::PrimitiveType_SpaceToBatchND:
      return NewPrimitiveC<SpaceToBatchND>(primitive);
    case schema::PrimitiveType_BroadcastTo:
      return NewPrimitiveC<BroadcastTo>(primitive);
    case schema::PrimitiveType_DepthToSpace:
      return NewPrimitiveC<DepthToSpace>(primitive);
    case schema::PrimitiveType_Lstm:
      return NewPrimitiveC<Lstm>(primitive);
    case schema::PrimitiveType_ZerosLike:
      return NewPrimitiveC<ZerosLike>(primitive);
    case schema::PrimitiveType_MakeTuple:
      return NewPrimitiveC<MakeTuple>(primitive);
    case schema::PrimitiveType_Where:
      return NewPrimitiveC<Where>(primitive);
    case schema::PrimitiveType_ScatterND:
      return NewPrimitiveC<ScatterND>(primitive);
    case schema::PrimitiveType_ConstantOfShape:
      return NewPrimitiveC<ConstantOfShape>(primitive);
    case schema::PrimitiveType_L2Norm:
      return NewPrimitiveC<L2Norm>(primitive);
    case schema::PrimitiveType_SparseToDense:
      return NewPrimitiveC<SparseToDense>(primitive);
    default:
      MS_LOG(ERROR) << "Unsupported primitive type in UnPackFromSchemaPrimitive : "
                    << schema::EnumNamePrimitiveType(op_type);
      break;
  }
  return nullptr;
}
#endif

int PrimitiveC::Type() const {
  if (this->primitive_ == nullptr) {
    return schema::PrimitiveType_NONE;
  }
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
