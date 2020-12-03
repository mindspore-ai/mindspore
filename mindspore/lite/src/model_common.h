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

#ifndef MINDSPORE_LITE_SRC_MODEL_COMMON_H_
#define MINDSPORE_LITE_SRC_MODEL_COMMON_H_

#include <string>
#include "src/ops/primitive_c.h"
#include "include/model.h"
#include "include/version.h"
#include "schema/model_generated.h"
#include "schema/model_v0_generated.h"
#include "src/common/common.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore::lite {
int ConvertSubGraph(const schema::SubGraph &sub_graph, Model *model);

template <typename T = schema::MetaGraph, typename U = schema::CNode>
bool ConvertNodes(const T &meta_graph, Model *model, int schema_version = SCHEMA_CUR) {
  if (model == nullptr || meta_graph.nodes() == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is invalid, please check your model file.";
    return false;
  }
  for (size_t i = 0; i < meta_graph.nodes()->size(); ++i) {
    auto *node = new (std::nothrow) Model::Node();
    if (node == nullptr) {
      MS_LOG(ERROR) << "new node fail!";
      return false;
    }
    auto c_node = meta_graph.nodes()->template GetAs<U>(i);
    auto src_prim = reinterpret_cast<const schema::Primitive *>(c_node->primitive());
#ifdef PRIMITIVE_WRITEABLE
    node->primitive_ = PrimitiveC::Create(const_cast<schema::Primitive *>(src_prim));
#else
    auto primitive = const_cast<schema::Primitive *>(src_prim);
    auto func_pointer = OpsRegistry::GetInstance()->GetPrimitiveCreator(primitive->value_type());
    if (func_pointer == nullptr) {
      MS_LOG(ERROR) << "PrimitiveCreator function pointer is nullptr, type: "
                    << schema::EnumNamePrimitiveType(primitive->value_type());
      delete node;
      return false;
    }
    node->primitive_ = func_pointer(primitive);
#endif
    if (node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "unpack primitive == nullptr!";
      delete node;
      return false;
    }
    node->primitive_->set_quant_type(static_cast<schema::QuantType>(c_node->quantType()));
    node->name_ = c_node->name()->c_str();
    node->node_type_ = static_cast<NodeType>(c_node->nodeType());
    auto count = c_node->inputIndex()->size();
    for (uint32_t j = 0; j < count; ++j) {
      node->input_indices_.push_back(size_t(c_node->inputIndex()->template GetAs<uint32_t>(j)));
    }
    if (c_node->outputIndex() != nullptr) {
      count = c_node->outputIndex()->size();
      for (uint32_t j = 0; j < count; ++j) {
        node->output_indices_.push_back(size_t(c_node->outputIndex()->template GetAs<uint32_t>(j)));
      }
    }
    model->all_nodes_.push_back(node);
  }
  return true;
}

template <typename T = schema::MetaGraph>
bool ConvertTensors(const T &meta_graph, Model *model) {
  if (model == nullptr || meta_graph.allTensors() == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is invalid, please check your model file.";
    return false;
  }
  auto tensor_count = meta_graph.allTensors()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *tensor = meta_graph.allTensors()->template GetAs<schema::Tensor>(i);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return false;
    }
    model->all_tensors_.push_back(const_cast<mindspore::schema::Tensor *>(tensor));
  }
  return true;
}

template <typename T = schema::MetaGraph>
int MetaGraphMappingSubGraph(const T &meta_graph, Model *model) {
  if (model == nullptr || meta_graph.inputIndex() == nullptr || meta_graph.outputIndex() == nullptr ||
      meta_graph.nodes() == nullptr || meta_graph.allTensors() == nullptr) {
    MS_LOG(ERROR) << "model or meta_graph is invalid, please check your model file.";
    return RET_ERROR;
  }
  auto *subgraph = new (std::nothrow) Model::SubGraph();
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }
  if (meta_graph.name() != nullptr) {
    subgraph->name_ = meta_graph.name()->c_str();
  }
  auto in_count = meta_graph.inputIndex()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    subgraph->input_indices_.push_back(size_t(meta_graph.inputIndex()->template GetAs<uint32_t>(i)));
  }
  auto out_count = meta_graph.outputIndex()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    subgraph->output_indices_.push_back(size_t(meta_graph.outputIndex()->template GetAs<uint32_t>(i)));
  }
  auto node_count = meta_graph.nodes()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    subgraph->node_indices_.push_back(i);
  }
  auto tensor_count = meta_graph.allTensors()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    subgraph->tensor_indices_.push_back(i);
  }
  model->sub_graphs_.push_back(subgraph);
  return RET_OK;
}

template <typename T = schema::MetaGraph, typename U = schema::CNode>
int GenerateModel(const T &meta_graph, Model *model, int schema_version = 0) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return RET_ERROR;
  }
  if (meta_graph.name() != nullptr) {
    model->name_ = meta_graph.name()->c_str();
  }
  if (meta_graph.version() != nullptr) {
    model->version_ = meta_graph.version()->c_str();
  }
  if (!ConvertNodes<T, U>(meta_graph, model, schema_version)) {
    MS_LOG(ERROR) << "convert node failed";
    return RET_ERROR;
  }
  if (!ConvertTensors<T>(meta_graph, model)) {
    MS_LOG(ERROR) << "convert tensor failed";
    return RET_ERROR;
  }
  if (meta_graph.subGraph() == nullptr) {
    int ret = MetaGraphMappingSubGraph<T>(meta_graph, model);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "converter old version model wrong.";
      return ret;
    }
  } else {
    auto sub_graphs = meta_graph.subGraph();
    auto sub_graph_size = sub_graphs->size();
    for (size_t i = 0; i < sub_graph_size; i++) {
      auto sub_graph = sub_graphs->template GetAs<schema::SubGraph>(i);
      int ret = ConvertSubGraph(*sub_graph, model);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "converter subgraph wrong.";
        return ret;
      }
    }
  }
  return RET_OK;
}

int VersionVerify(flatbuffers::Verifier *verify);

int NodeVerify(const Model &model);

int SubGraphVerify(const Model &model);

int ModelVerify(const Model &model, const int &schema_version);

const void *GetMetaGraphByVerison(const char *buf, const int &schema_version);

int GenerateModelByVersion(const void *meta_graph, Model *model, const int &schema_version);

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_MODEL_COMMON_H_
