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

#include "src/ops/unique.h"
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
#include "src/ops/primitive_c.h"
#include "include/model.h"
#include "utils/log_adapter.h"

namespace mindspore::lite {

class ModelImpl {
 public:
  static ModelImpl *Import(const char *model_buf, size_t size);
  ModelImpl() = default;
  explicit ModelImpl(const char *model_buf, size_t size) : model_buf_(model_buf), buf_size_(size) {
    meta_graph_ = schema::GetMetaGraph(model_buf);
  }
  virtual ~ModelImpl();
  PrimitiveC *GetOp(const std::string &name) const;
  const schema::MetaGraph *meta_graph() const;
  void FreeMetaGraph();
  int BuildOps();

 protected:
  PrimitiveC *CopyPrimitive(const schema::Primitive *src_prim);

 protected:
  const char *model_buf_;
  size_t buf_size_;
  const schema::MetaGraph *meta_graph_ = nullptr;
  std::map<std::string, PrimitiveC *> ops_;
};

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

PrimitiveC *ModelImpl::GetOp(const std::string &name) const {
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

PrimitiveC *ModelImpl::CopyPrimitive(const schema::Primitive *src_prim) {
  MS_EXCEPTION_IF_NULL(src_prim);
  auto op_type = src_prim->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new SoftMax(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Activation:
      return new Activation(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Conv2D:
      return new Conv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DeConv2D:
      return new DeConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reduce:
      return new Reduce(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Pooling:
      return new Pooling(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DepthwiseConv2D:
      return new DepthwiseConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FusedBatchNorm:
      return new FusedBatchNorm(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_BatchNorm:
      return new BatchNorm(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FullConnection:
      return new FullConnection(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Power:
      return new Power(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Range:
      return new Range(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Mul:
      return new Mul(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Add:
      return new Add(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sub:
      return new Sub(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Div:
      return new Div(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_BiasAdd:
      return new BiasAdd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ExpandDims:
      return new ExpandDims(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ArgMax:
      return new ArgMax(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ArgMin:
      return new ArgMin(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Cast:
      return new Cast(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reshape:
      return new Reshape(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Scale:
      return new Scale(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Eltwise:
      return new Eltwise(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Concat:
      return new Concat(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Fill:
      return new Fill(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Transpose:
      return new Transpose(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Slice:
      return new SliceOp(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Squeeze:
      return new Squeeze(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Nchw2Nhwc:
      return new Nchw2Nhwc(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Nhwc2Nchw:
      return new Nhwc2Nchw(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Flatten:
      return new Flatten(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Mean:
      return new Mean(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Stack:
      return new Stack(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Crop:
      return new Crop(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_SquaredDifference:
      return new SquaredDifference(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_AddN:
      return new AddN(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Abs:
      return new Abs(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sin:
      return new Sin(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Cos:
      return new Cos(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Log:
      return new Log(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Sqrt:
      return new Sqrt(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Rsqrt:
      return new Rsqrt(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Square:
      return new Square(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Exp:
      return new Exp(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Gather:
      return new Gather(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_GatherNd:
      return new GatherNd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LocalResponseNormalization:
      return new LocalResponseNormalization(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Maximum:
      return new Maximum(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Minimum:
      return new Minimum(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Pad:
      return new Pad(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_StridedSlice:
      return new StridedSlice(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Prelu:
      return new Prelu(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_CaffePReLU:
      return new CaffePReLU(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Round:
      return new Round(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Reverse:
      return new Reverse(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_ReverseSequence:
      return new ReverseSequence(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalAnd:
      return new LogicalAnd(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalOr:
      return new LogicalOr(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LogicalNot:
      return new LogicalNot(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FloorDiv:
      return new FloorDiv(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_FloorMod:
      return new FloorMod(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Equal:
      return new Equal(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_NotEqual:
      return new NotEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Less:
      return new Less(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_LessEqual:
      return new LessEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Greater:
      return new Greater(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_GreaterEqual:
      return new GreaterEqual(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Floor:
      return new Floor(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Ceil:
      return new Ceil(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Split:
      return new Split(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_OneHot:
      return new OneHot(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_SpaceToDepth:
      return new SpaceToDepth(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Tile:
      return new Tile(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Resize:
      return new Resize(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Unstack:
      return new Unstack(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Unique:
      return new Unique(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_TopK:
      return new TopK(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_MatMul:
      return new MatMul(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_QuantDTypeCast:
      return new QuantDTypeCast(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_EmbeddingLookup:
      return new EmbeddingLookup(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Elu:
      return new Elu(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return new DeDepthwiseConv2D(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Shape:
      return new Shape(const_cast<schema::Primitive *>(src_prim));
    case schema::PrimitiveType_Unsqueeze:
      return new Unsqueeze(const_cast<schema::Primitive *>(src_prim));
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

Model *Model::Import(const char *model_buf, size_t size) {
  auto model = new Model();
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "model buf is null";
    return nullptr;
  }
  model->model_impl_ = ModelImpl::Import(model_buf, size);
  if (model->model_impl_ == nullptr) {
    MS_LOG(ERROR) << "model impl is null";
    return nullptr;
  }
  return model;
}

Model::~Model() { delete (this->model_impl_); }

mindspore::lite::PrimitiveC *Model::GetOp(const std::string &name) const {
  MS_EXCEPTION_IF_NULL(model_impl_);
  return const_cast<PrimitiveC *>(model_impl_->GetOp(name));
}

void Model::FreeMetaGraph() {
  MS_EXCEPTION_IF_NULL(model_impl_);
  return model_impl_->FreeMetaGraph();
}

const schema::MetaGraph *Model::GetMetaGraph() const {
  MS_EXCEPTION_IF_NULL(model_impl_);
  return model_impl_->meta_graph();
}

}  // namespace mindspore::lite
