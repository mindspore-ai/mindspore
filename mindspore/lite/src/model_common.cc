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
#include "src/ops/while.h"

namespace mindspore::lite {
int ConvertSubGraph(const schema::SubGraph &sub_graph, Model *model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is null.";
    return RET_ERROR;
  }
  if (sub_graph.name() == nullptr || sub_graph.inputIndices() == nullptr || sub_graph.outputIndices() == nullptr ||
      sub_graph.nodeIndices() == nullptr || sub_graph.tensorIndices() == nullptr) {
    MS_LOG(ERROR) << "sub_graph is invalid.";
    return RET_ERROR;
  }

  auto *subgraph = new (std::nothrow) Model::SubGraph();
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }

  subgraph->name_ = sub_graph.name()->c_str();
  auto in_count = sub_graph.inputIndices()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    subgraph->input_indices_.push_back(sub_graph.inputIndices()->Get(i));
  }
  auto out_count = sub_graph.outputIndices()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    subgraph->output_indices_.push_back(sub_graph.outputIndices()->Get(i));
  }
  auto node_count = sub_graph.nodeIndices()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    subgraph->node_indices_.push_back(sub_graph.nodeIndices()->Get(i));
  }
  auto tensor_count = sub_graph.tensorIndices()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    subgraph->tensor_indices_.push_back(sub_graph.tensorIndices()->Get(i));
  }
  model->sub_graphs_.push_back(subgraph);
  return RET_OK;
}

int VersionVerify(flatbuffers::Verifier *verify) {
  if (verify == nullptr) {
    MS_LOG(ERROR) << "verify is null.";
    return RET_ERROR;
  }
  if (schema::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_CUR;
  } else if (schema::v0::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_V0;
  }
  return SCHEMA_VERSION::SCHEMA_INVALID;
}

int NodeVerify(const Model &model) {
  auto tensor_size = model.all_tensors_.size();
  uint32_t subGraph_size = model.sub_graphs_.size();

  for (auto &node : model.all_nodes_) {
    if (node == nullptr || node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "node or its primitive_ is null.";
      return RET_ERROR;
    }
    if (std::any_of(node->input_indices_.begin(), node->input_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->input_indices_ is beyond size.";
      return RET_ERROR;
    }
    if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->output_indices_ is beyond size.";
      return RET_ERROR;
    }

    auto prim = node->primitive_;
    if (prim->Type() == schema::PrimitiveType_While) {
      auto whileOp = reinterpret_cast<mindspore::lite::While *>(const_cast<mindspore::lite::PrimitiveC *>(prim));
      if (whileOp == nullptr) {
        MS_LOG(ERROR) << "whileOp is null.";
        return RET_ERROR;
      }
      if (static_cast<uint32_t>(whileOp->GetBodySubgraphIndex()) >= subGraph_size ||
          static_cast<uint32_t>(whileOp->GetCondSubgraphIndex()) >= subGraph_size) {
        MS_LOG(ERROR) << "index of subGraph is beyond subGraph_size.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int SubGraphVerify(const Model &model) {
  auto tensor_size = model.all_tensors_.size();
  auto node_size = model.all_nodes_.size();

  for (auto &graph : model.sub_graphs_) {
    if (graph == nullptr) {
      MS_LOG(ERROR) << "graph is null.";
      return RET_ERROR;
    }
    if (std::any_of(graph->input_indices_.begin(), graph->input_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->input_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->output_indices_.begin(), graph->output_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->output_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->tensor_indices_.begin(), graph->tensor_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->tensor_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->node_indices_.begin(), graph->node_indices_.end(),
                    [&node_size](const uint32_t &idx) { return idx >= node_size; })) {
      MS_LOG(ERROR) << "Index of graph->node_indices_ is beyond node_size.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ModelVerify(const Model &model, const int &schema_version) {
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    return NodeVerify(model) == RET_OK && SubGraphVerify(model) == RET_OK;
  } else if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    return NodeVerify(model) == RET_OK;
  }
  return RET_ERROR;
}

const void *GetMetaGraphByVerison(const char *buf, const int &schema_version) {
  if (buf == nullptr) {
    MS_LOG(ERROR) << "buf is null.";
    return nullptr;
  }
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    return reinterpret_cast<const void *>(schema::GetMetaGraph(buf));
  } else if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    return reinterpret_cast<const void *>(schema::v0::GetMetaGraph(buf));
  }
  return nullptr;
}

int GenerateModelByVersion(const void *meta_graph, Model *model, const int &schema_version) {
  if (meta_graph == nullptr || model == nullptr) {
    MS_LOG(ERROR) << "meta_graph or model is null.";
    return RET_ERROR;
  }
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

  return ModelVerify(*model, schema_version) ? model : nullptr;
}
}  // namespace mindspore::lite
