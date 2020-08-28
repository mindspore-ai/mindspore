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
  MS_ASSERT(nullptr != meta_graph_->nodes());
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
  MS_ASSERT(nullptr != model_impl_);
  return const_cast<PrimitiveC *>(model_impl_->GetOp(name));
}

void Model::FreeMetaGraph() {
  MS_ASSERT(nullptr != model_impl_);
  model_impl_->FreeMetaGraph();
}

const schema::MetaGraph *Model::GetMetaGraph() const {
  MS_ASSERT(nullptr != model_impl_);
  return model_impl_->meta_graph();
}

}  // namespace mindspore::lite
