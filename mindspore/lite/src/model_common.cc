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
#include "src/model_common.h"
#include "include/version.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore::lite {
bool ConvertNodes(const schema::MetaGraph *meta_graph, Model *model) {
  for (size_t i = 0; i < meta_graph->nodes()->size(); ++i) {
    auto *node = new (std::nothrow) Model::Node();
    if (node == nullptr) {
      MS_LOG(ERROR) << "new node fail!";
      return false;
    }
    auto c_node = meta_graph->nodes()->GetAs<schema::CNode>(i);
    auto src_prim = c_node->primitive();
#ifdef PRIMITIVE_WRITEABLE
    node->primitive_ = PrimitiveC::Create(const_cast<schema::Primitive *>(src_prim));
#else
    auto primitive = const_cast<schema::Primitive *>(src_prim);
    node->primitive_ = OpsRegistry::GetInstance()->getPrimitiveCreator(primitive->value_type())(primitive);
#endif
    if (node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "unpack primitive == nullptr!";
      delete node;
      return false;
    }
    node->primitive_->SetQuantType(c_node->quantType());
    node->name_ = c_node->name()->c_str();
    node->node_type_ = c_node->nodeType();
    auto count = c_node->inputIndex()->size();
    for (uint32_t j = 0; j < count; ++j) {
      node->input_indices_.push_back(size_t(c_node->inputIndex()->GetAs<uint32_t>(j)));
    }
    if (c_node->outputIndex() != nullptr) {
      count = c_node->outputIndex()->size();
      for (uint32_t j = 0; j < count; ++j) {
        node->output_indices_.push_back(size_t(c_node->outputIndex()->GetAs<uint32_t>(j)));
      }
    }
    model->nodes_.push_back(node);
  }
  return true;
}

bool ConvertTensors(const schema::MetaGraph *meta_graph, Model *model) {
  auto tensor_count = meta_graph->allTensors()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(i);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return false;
    }
    model->all_tensors_.push_back(const_cast<mindspore::schema::Tensor *>(tensor));
  }
  return true;
}

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  auto *model = new (std::nothrow) Model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }
  if (take_buf) {
    model->buf = const_cast<char *>(model_buf);
  } else {
    if (size == 0) {
      MS_LOG(ERROR) << "malloc size is equal to 0";
      delete (model);
      return nullptr;
    }
    model->buf = reinterpret_cast<char *>(malloc(size));
    if (model->buf == nullptr) {
      MS_LOG(ERROR) << "new inner model buf fail!";
      delete (model);
      return nullptr;
    }
    memcpy(model->buf, model_buf, size);
  }

  auto meta_graph = schema::GetMetaGraph(model->buf);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is nullptr!";
    delete (model);
    return nullptr;
  }

  if (meta_graph->name() != nullptr) {
    model->name_ = meta_graph->name()->c_str();
  }
  if (meta_graph->version() != nullptr) {
    model->version_ = meta_graph->version()->c_str();
  }

  if (model->version_ != Version()) {
    MS_LOG(WARNING) << "model version is " << model->version_ << ", inference version is " << Version() << " not equal";
  }

  auto in_count = meta_graph->inputIndex()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    model->input_indices_.push_back(size_t(meta_graph->inputIndex()->GetAs<uint32_t>(i)));
  }

  auto out_count = meta_graph->outputIndex()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    model->output_indices_.push_back(size_t(meta_graph->outputIndex()->GetAs<uint32_t>(i)));
  }
  if (!ConvertNodes(meta_graph, model)) {
    delete model;
    return nullptr;
  }

  if (!ConvertTensors(meta_graph, model)) {
    delete model;
    return nullptr;
  }
  return model;
}
}  // namespace mindspore::lite
