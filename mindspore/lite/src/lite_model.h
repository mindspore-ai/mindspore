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

#ifndef MINDSPORE_LITE_SRC_LITE_MODEL_H_
#define MINDSPORE_LITE_SRC_LITE_MODEL_H_

#include <string>
#include <vector>
#include "include/errorcode.h"
#include "include/model.h"
#include "include/version.h"
#include "schema/model_generated.h"
#include "src/common/common.h"
#include "src/common/version_manager.h"
#ifdef ENABLE_V0
#include "schema/model_v0_generated.h"
#endif

namespace mindspore {
namespace lite {
class LiteModel : public Model {
 public:
  int ConstructModel();

  bool ModelVerify() const;

  void Free() override;

  void Destroy() override;

  ~LiteModel() override { Destroy(); }

 private:
#ifdef ENABLE_V0
  int ConvertAttrs(Model::Node *node, std::vector<schema::Tensor *> *dst_tensor);

  int ConvertAttrToTensors();
#endif

  template <typename T = schema::MetaGraph, typename U = schema::CNode>
  bool ConvertNodes(const T &meta_graph) {
    if (meta_graph.nodes() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
      return false;
    }
    for (size_t i = 0; i < meta_graph.nodes()->size(); ++i) {
      auto *node = new (std::nothrow) Model::Node();
      if (node == nullptr) {
        MS_LOG(ERROR) << "new node fail!";
        return false;
      }
      auto c_node = meta_graph.nodes()->template GetAs<U>(i);
      node->primitive_ = c_node->primitive();
      node->quant_type_ = c_node->quantType();
      node->name_ = c_node->name()->c_str();
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
      this->all_nodes_.push_back(node);
    }
    return true;
  }

  template <typename T = schema::MetaGraph>
  bool ConvertTensors(const T &meta_graph) {
    if (meta_graph.allTensors() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
      return false;
    }
    auto tensor_count = meta_graph.allTensors()->size();
    for (uint32_t i = 0; i < tensor_count; ++i) {
      auto *tensor = meta_graph.allTensors()->template GetAs<schema::Tensor>(i);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << i << "the tensor in metagraph is nullptr";
        return false;
      }
      this->all_tensors_.push_back(const_cast<mindspore::schema::Tensor *>(tensor));
    }
    return true;
  }

  template <typename T = schema::MetaGraph>
  int MetaGraphMappingSubGraph(const T &meta_graph) {
    if (meta_graph.inputIndex() == nullptr || meta_graph.outputIndex() == nullptr || meta_graph.nodes() == nullptr ||
        meta_graph.allTensors() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
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
    this->sub_graphs_.push_back(subgraph);
    return RET_OK;
  }

  template <typename T = schema::MetaGraph, typename U = schema::CNode>
  int GenerateModel(const T &meta_graph) {
    if (meta_graph.name() != nullptr) {
      this->name_ = meta_graph.name()->c_str();
    }
    if (meta_graph.version() != nullptr) {
      this->version_ = meta_graph.version()->c_str();
    }
    if (!ConvertNodes<T, U>(meta_graph)) {
      MS_LOG(ERROR) << "convert node failed";
      return RET_ERROR;
    }
    if (!ConvertTensors<T>(meta_graph)) {
      MS_LOG(ERROR) << "convert tensor failed";
      return RET_ERROR;
    }
    if (meta_graph.subGraph() == nullptr) {
      int ret = MetaGraphMappingSubGraph<T>(meta_graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "converter old version model wrong.";
        return ret;
      }
    } else {
      auto sub_graphs = meta_graph.subGraph();
      MS_ASSERT(sub_graphs != nullptr);
      auto sub_graph_size = sub_graphs->size();
      for (size_t i = 0; i < sub_graph_size; i++) {
        auto sub_graph = sub_graphs->template GetAs<schema::SubGraph>(i);
        int ret = ConvertSubGraph(*sub_graph);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "converter subgraph wrong.";
          return ret;
        }
      }
    }
#ifdef ENABLE_V0
    if (ConvertAttrToTensors() != RET_OK) {
      MS_LOG(ERROR) << "fail to convert attr to tensor.";
      return RET_ERROR;
    }
#endif
    return RET_OK;
  }

  int VersionVerify(flatbuffers::Verifier *verify) const;

  const void *GetMetaGraphByVerison();

  int GenerateModelByVersion(const void *meta_graph);

  int ConvertSubGraph(const schema::SubGraph &sub_graph);

  int NodeVerify() const;

  int SubGraphVerify() const;

 public:
  size_t buf_size_ = 0;

 protected:
  std::vector<char *> attr_tensor_bufs_;
};

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_LITE_MODEL_H_
