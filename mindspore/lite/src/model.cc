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
#include "include/model.h"
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

    this->ops_[name] = PrimitiveC::UnPackFromSchemaPrimitive(const_cast<schema::Primitive *>(srcPrim));
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
