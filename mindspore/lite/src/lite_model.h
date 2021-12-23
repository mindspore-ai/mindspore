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
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "include/model.h"
#include "include/version.h"
#include "schema/model_generated.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/schema_tensor_wrapper.h"
#include "nnacl/op_base.h"
#include "src/common/prim_util.h"
#ifdef ENABLE_V0
#include "schema/model_v0_generated.h"
#endif
#ifdef ENABLE_MODEL_OBF
#include "tools/obfuscator/include/deobfuscator.h"
#endif

namespace mindspore {
namespace lite {
class LiteModel : public Model {
 public:
  explicit LiteModel(std::string model_path = "") : model_path_(std::move(model_path)) {}

  int ConstructModel(const char *model_buf, size_t size, bool take_buf);

  bool ModelVerify() const;

  void Free() override;

  void Destroy() override;

  ~LiteModel() override { Destroy(); }

  bool keep_model_buf() const { return this->keep_model_buf_; }

  void set_keep_model_buf(bool keep) { this->keep_model_buf_ = keep; }

  int GetSchemaVersion() const { return schema_version_; }

  SchemaTensorWrapper *GetSchemaTensor(const size_t &tensor_index) const;

  static int VersionVerify(flatbuffers::Verifier *verify);

 private:
#ifdef ENABLE_V0
  int ConvertAttrs(Model::Node *node, std::vector<schema::Tensor *> *dst_tensor);

  int ConvertAttrToTensors();
#endif

  bool PrepareInnerTensors();

  template <typename T = schema::MetaGraph, typename U = schema::CNode>
  bool ConvertNodes(const T &meta_graph) {
    MS_CHECK_TRUE_MSG(meta_graph.nodes() != nullptr, false, "meta_graph is invalid, please check your model file.");
    for (size_t i = 0; i < meta_graph.nodes()->size(); ++i) {
      auto *node = new (std::nothrow) Model::Node();
      MS_CHECK_TRUE_MSG(node != nullptr, false, "new node fail!");
      auto c_node = meta_graph.nodes()->template GetAs<U>(i);
      MS_CHECK_TRUE_MSG(c_node != nullptr, false, "get as cnode fail!");
      node->node_type_ = GetPrimitiveType(c_node->primitive(), schema_version_);
#ifdef ENABLE_MODEL_OBF
      auto src_prim = reinterpret_cast<const schema::Primitive *>(c_node->primitive());
      auto src_prim_type = src_prim->value_type();
      unsigned char *dst_prim = nullptr;
      if (src_prim_type == schema::PrimitiveType_GenOP) {
        auto src_node_stat = this->all_nodes_stat_[i];
        auto dst_prim_type = this->all_prims_type_[i];
        auto ret = DeObfuscatePrimitive(src_prim, src_node_stat, &dst_prim, schema::PrimitiveType(dst_prim_type));
        if (!ret) {
          MS_LOG(ERROR) << "Deobfuscate primitive failed!";
          delete node;
          return false;
        }
        if (dst_prim == nullptr) {
          this->all_nodes_.push_back(node);
          continue;
        }
        this->deobf_prims_.push_back(dst_prim);
        src_prim = reinterpret_cast<const schema::Primitive *>(flatbuffers::GetRoot<schema::Primitive>(dst_prim));
      }
      node->primitive_ = const_cast<schema::Primitive *>(src_prim);
#else
      node->primitive_ = c_node->primitive();
#endif
      node->quant_type_ = c_node->quantType();
      if (node->quant_type_ < schema::QuantType_MIN || node->quant_type_ > schema::QuantType_MAX) {
        MS_LOG(ERROR) << "node->quant_type_:" << node->quant_type_ << " is invalid.";
        delete node;
        return false;
      }
      if (schema_version_ == SCHEMA_VERSION::SCHEMA_CUR) {
        SetNodeDeviceType(node, *c_node);
      }
#ifdef ENABLE_V0
      if (schema_version_ == SCHEMA_VERSION::SCHEMA_V0) {
        SetNodeDeviceType(node, *c_node);
      }
#endif
      if (node->quant_type_ == schema::QuantType_PostTraining || node->quant_type_ == schema::QuantType_AwareTraining) {
        node->quant_type_ = schema::QuantType_QUANT_ALL;
      } else if (node->quant_type_ == schema::QuantType_WeightQuant) {
        node->quant_type_ = schema::QuantType_QUANT_WEIGHT;
      }
      if (c_node->name() == nullptr) {
        node->name_ = "";
      } else {
        node->name_ = c_node->name()->c_str();
      }
      if (c_node->inputIndex() != nullptr) {
        auto count = c_node->inputIndex()->size();
        for (uint32_t j = 0; j < count; ++j) {
          node->input_indices_.push_back(size_t(c_node->inputIndex()->template GetAs<uint32_t>(j)));
        }
      }
      if (c_node->outputIndex() != nullptr) {
        auto count = c_node->outputIndex()->size();
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

    if (meta_graph.inputIndex() == nullptr || meta_graph.outputIndex() == nullptr ||
        meta_graph.allTensors() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
      return RET_ERROR;
    }

    // converterInputOutput
    auto in_count = meta_graph.inputIndex()->size();
    for (uint32_t i = 0; i < in_count; ++i) {
      this->input_indices_.push_back(meta_graph.inputIndex()->Get(i));
    }
    auto out_count = meta_graph.outputIndex()->size();
    for (uint32_t i = 0; i < out_count; ++i) {
      this->output_indices_.push_back(meta_graph.outputIndex()->Get(i));
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

  void SetNodeDeviceType(Node *node, const schema::CNode &c_node) { node->device_type_ = c_node.deviceType(); }

#ifdef ENABLE_V0
  void SetNodeDeviceType(Node *node, const schema::v0::CNode &c_node) { node->device_type_ = -1; }
#endif

  int GenerateModelByVersion();

  int ConvertSubGraph(const schema::SubGraph &sub_graph);

  int NodeVerify() const;

  int SubGraphVerify() const;

  int SubGraphInOutVerify(const Model::SubGraph *graph) const;

 public:
  size_t buf_size_ = 0;
  std::vector<void *> node_bufs_;

 protected:
  std::vector<char *> attr_tensor_bufs_;
  bool keep_model_buf_ = false;
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
  // tensor_index --- external_data
  std::vector<SchemaTensorWrapper *> inner_all_tensors_;
  const std::string model_path_;
};

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf);
LiteModel *LiteImportFromPath(const char *model_path);
Model *ImportFromPath(const char *model_path);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_LITE_MODEL_H_
