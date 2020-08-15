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

#include <memory>
#include <string>
#include "src/model_impl.h"
#include "utils/log_adapter.h"

namespace mindspore::lite {
ModelImpl *ModelImpl::Import(const char *model_buf, size_t size) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  auto *inner_model_buf = new (std::nothrow) char[size];
  if (inner_model_buf == nullptr) {
    MS_LOG(ERROR) << "new model buf fail.";
    return nullptr;
  }
  memcpy(inner_model_buf, model_buf, size);
  auto model = new (std::nothrow) ModelImpl(inner_model_buf, size);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Create modelImpl failed";
    return nullptr;
  }
  auto ret = model->BuildOps();
  if (0 != ret) {
    MS_LOG(ERROR) << "BuildOps failed";
    return nullptr;
  }
  return model;
}

lite::Primitive *ModelImpl::GetOp(const std::string &name) const {
  auto iter = ops_.find(name);
  if (iter == ops_.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

ModelImpl::~ModelImpl() {
  delete[](this->model_buf_);
  for (auto iter : ops_) {
    delete (iter.second);
  }
  ops_.clear();
}

void ModelImpl::FreeMetaGraph() {
  delete[](this->model_buf_);
  model_buf_ = nullptr;
}

const schema::MetaGraph *ModelImpl::meta_graph() const { return this->meta_graph_; }

lite::Primitive *ModelImpl::CopyPrimitive(const schema::Primitive *src_prim) {
  MS_EXCEPTION_IF_NULL(src_prim);
  auto op_type = src_prim->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new lite::SoftMax(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Activation:
      return new lite::Activation(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Conv2D:
      return new lite::Conv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DeConv2D:
      return new lite::DeConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reduce:
      return new lite::Reduce(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Pooling:
      return new lite::Pooling(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DepthwiseConv2D:
      return new lite::DepthwiseConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FusedBatchNorm:
      return new lite::FusedBatchNorm(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_BatchNorm:
      return new lite::BatchNorm(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FullConnection:
      return new lite::FullConnection(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Power:
      return new lite::Power(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Range:
      return new lite::Range(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Mul:
      return new lite::Mul(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Add:
      return new lite::Add(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sub:
      return new lite::Sub(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Div:
      return new lite::Div(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_BiasAdd:
      return new lite::BiasAdd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ExpandDims:
      return new lite::ExpandDims(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ArgMax:
      return new lite::ArgMax(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ArgMin:
      return new lite::ArgMin(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Cast:
      return new lite::Cast(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reshape:
      return new lite::Reshape(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Scale:
      return new lite::Scale(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Eltwise:
      return new lite::Eltwise(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Concat:
      return new lite::Concat(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Fill:
      return new lite::Fill(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Transpose:
      return new lite::Transpose(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Slice:
      return new lite::Slice(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Squeeze:
      return new lite::Squeeze(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Nchw2Nhwc:
      return new lite::Nchw2Nhwc(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Nhwc2Nchw:
      return new lite::Nhwc2Nchw(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Flatten:
      return new lite::Flatten(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Mean:
      return new lite::Mean(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Stack:
      return new lite::Stack(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Crop:
      return new lite::Crop(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_SquaredDifference:
      return new lite::SquaredDifference(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_AddN:
      return new lite::AddN(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Abs:
      return new lite::Abs(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sin:
      return new lite::Sin(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Cos:
      return new lite::Cos(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Log:
      return new lite::Log(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sqrt:
      return new lite::Sqrt(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Rsqrt:
      return new lite::Rsqrt(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Square:
      return new lite::Square(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Exp:
      return new lite::Exp(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Gather:
      return new lite::Gather(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_GatherNd:
      return new lite::GatherNd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LocalResponseNormalization:
      return new lite::LocalResponseNormalization(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Maximum:
      return new lite::Maximum(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Minimum:
      return new lite::Minimum(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Pad:
      return new lite::Pad(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_StridedSlice:
      return new lite::StridedSlice(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Prelu:
      return new lite::Prelu(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_CaffePReLU:
      return new lite::CaffePReLU(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Round:
      return new lite::Round(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reverse:
      return new lite::Reverse(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ReverseSequence:
      return new lite::ReverseSequence(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalAnd:
      return new lite::LogicalAnd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalOr:
      return new lite::LogicalOr(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalNot:
      return new lite::LogicalNot(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FloorDiv:
      return new lite::FloorDiv(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FloorMod:
      return new lite::FloorMod(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Equal:
      return new lite::Equal(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_NotEqual:
      return new lite::NotEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Less:
      return new lite::Less(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LessEqual:
      return new lite::LessEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Greater:
      return new lite::Greater(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_GreaterEqual:
      return new lite::GreaterEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Floor:
      return new lite::Floor(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Ceil:
      return new lite::Ceil(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Split:
      return new lite::Split(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_OneHot:
      return new lite::OneHot(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_SpaceToDepth:
      return new lite::SpaceToDepth(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Tile:
      return new lite::Tile(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Resize:
      return new lite::Resize(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Unstack:
      return new lite::Unstack(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Unique:
      return new lite::Unique(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_TopK:
      return new lite::TopK(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_MatMul:
      return new lite::MatMul(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_QuantDTypeCast:
      return new lite::QuantDTypeCast(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_EmbeddingLookup:
      return new lite::EmbeddingLookup(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Elu:
      return new lite::Elu(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return new lite::DeconvDepthwiseConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Shape:
      return new lite::Shape(const_cast<schema::Primitive *>(src_prim));
    default:
      break;
  }
  return nullptr;
}

int ModelImpl::BuildOps() {
  if (this->meta_graph_ == nullptr) {
    MS_LOG(ERROR) << "mete_graph is nullptr";
    return -1;
  }
  MS_EXCEPTION_IF_NULL(meta_graph_->nodes());
  for (size_t i = 0; i < meta_graph_->nodes()->size(); i++) {
    auto cNode = meta_graph_->nodes()->GetAs<schema::CNode>(i);
    auto name = cNode->name()->str();
    auto srcPrim = cNode->primitive();

    this->ops_[name] = CopyPrimitive(srcPrim);
    //    flatbuffers::FlatBufferBuilder fbb(1024);
    //    schema::Conv2DBuilder conv2DBuilder(fbb);
    //    conv2DBuilder.add_padMode(srcPrim->value_as_Conv2D()->padMode());
    //    conv2DBuilder.add_channelOut(srcPrim->value_as_Conv2D()->channelOut());
    //    conv2DBuilder.add_channelIn(srcPrim->value_as_Conv2D()->channelIn());
    //    conv2DBuilder.add_strideH(srcPrim->value_as_Conv2D()->strideH());
    //    conv2DBuilder.add_strideW(srcPrim->value_as_Conv2D()->strideW());
    //    conv2DBuilder.add_dilateH(srcPrim->value_as_Conv2D()->dilateH());
    //    conv2DBuilder.add_dilateW(srcPrim->value_as_Conv2D()->dilateW());
    //    conv2DBuilder.add_kernelH(srcPrim->value_as_Conv2D()->kernelH());
    //    conv2DBuilder.add_kernelW(srcPrim->value_as_Conv2D()->kernelW());
    //    conv2DBuilder.add_padUp(srcPrim->value_as_Conv2D()->padUp());
    //    conv2DBuilder.add_padDown(srcPrim->value_as_Conv2D()->padDown());
    //    conv2DBuilder.add_padLeft(srcPrim->value_as_Conv2D()->padLeft());
    //    conv2DBuilder.add_padRight(srcPrim->value_as_Conv2D()->padRight());
    //    conv2DBuilder.add_format(srcPrim->value_as_Conv2D()->format());
    //    conv2DBuilder.add_group(srcPrim->value_as_Conv2D()->group());
    //    conv2DBuilder.add_activationType(srcPrim->value_as_Conv2D()->activationType());
    //    schema::PrimitiveBuilder primBuilder(fbb);
    //    primBuilder.add_value_type(srcPrim->value_type());
    //    primBuilder.add_value(conv2DBuilder.Finish());
    //
    //    fbb.Finish(conv2DBuilder.Finish());
    //    auto buf = fbb.GetBufferPointer();
    //    auto conv2D = flatbuffers::GetRoot<schema::Conv2D>(buf);
    //    fbb.Clear();
    //
    //    return const_cast<mindspore::predict::OpDef *>(opDef);
  }
  return 0;
}
}  // namespace mindspore::lite
