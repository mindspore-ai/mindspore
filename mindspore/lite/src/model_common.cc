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

namespace mindspore::lite {
int ConvertSubGraph(const schema::SubGraph &sub_graph, Model *model) {
  MS_ASSERT(model != nullptr);
  auto *subgraph = new (std::nothrow) Model::SubGraph();
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }
  MS_ASSERT(sub_graph.name() != nullptr);
  subgraph->name_ = sub_graph.name()->c_str();
  MS_ASSERT(sub_graph.inputIndices() != nullptr);
  auto in_count = sub_graph.inputIndices()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    subgraph->input_indices_.push_back(sub_graph.inputIndices()->Get(i));
  }
  MS_ASSERT(sub_graph.outputIndices() != nullptr);
  auto out_count = sub_graph.outputIndices()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    subgraph->output_indices_.push_back(sub_graph.outputIndices()->Get(i));
  }
  MS_ASSERT(sub_graph.nodeIndices() != nullptr);
  auto node_count = sub_graph.nodeIndices()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    subgraph->node_indices_.push_back(sub_graph.nodeIndices()->Get(i));
  }
  MS_ASSERT(sub_graph.tensorIndices() != nullptr);
  auto tensor_count = sub_graph.tensorIndices()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    subgraph->tensor_indices_.push_back(sub_graph.tensorIndices()->Get(i));
  }
  model->sub_graphs_.push_back(subgraph);
  return RET_OK;
}

int VersionVerify(flatbuffers::Verifier *verify) {
  if (schema::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_CUR;
  } else if (schema::v0::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_V0;
  }
  return SCHEMA_VERSION::SCHEMA_INVALID;
}

const void *GetMetaGraphByVerison(const char *buf, const int &schema_version) {
  MS_ASSERT(buf != nullptr);
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    return reinterpret_cast<const void *>(schema::GetMetaGraph(buf));
  } else if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    return reinterpret_cast<const void *>(schema::v0::GetMetaGraph(buf));
  }
  return nullptr;
}

int GenerateModelByVersion(const void *meta_graph, Model *model, const int &schema_version) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(model != nullptr);
  int status = RET_ERROR;
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    status = GenerateModel<schema::MetaGraph, schema::CNode>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph),
                                                             model, schema_version);
  } else if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    status = GenerateModel<schema::v0::MetaGraph, schema::v0::CNode>(
      *reinterpret_cast<const schema::v0::MetaGraph *>(meta_graph), model, schema_version);
  }
  return status;
}

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  int schema_version = VersionVerify(&verify);
  if (schema_version == -1) {
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
  const void *meta_graph = GetMetaGraphByVerison(model->buf, schema_version);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is nullptr!";
    delete (model);
    return nullptr;
  }

  int status = GenerateModelByVersion(meta_graph, model, schema_version);
  if (status != RET_OK) {
    delete (model);
    MS_LOG(ERROR) << "fail to generate model";
    return nullptr;
  }

  if (model->version_ != Version()) {
    MS_LOG(WARNING) << "model version is " << model->version_ << ", inference version is " << Version() << " not equal";
  }
  if (model->sub_graphs_.empty()) {
    delete (model);
    return nullptr;
  }
  return model;
}
}  // namespace mindspore::lite
