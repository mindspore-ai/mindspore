/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "extendrt/mindir_loader/mindir_model/mindir_model_loader.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "src/litert/kernel_registry.h"
#include "ops/primitive_c.h"

namespace mindspore::infer::mindir {
const char kNodeTypeConstant[] = "Constant";

AbstractBaseModel *MindirModelLoader::ImportModel(const char *model_buf, size_t size, bool take_buf) {
  this->model_ = new (std::nothrow) MindirModel();
  MS_CHECK_TRUE_MSG(this->model_ != nullptr, nullptr,
                    "MindirModelLoader: Import model failed: new mindir model failed.");
  this->model_->model_type_ = mindspore::lite::ModelType_MindIR;
  auto ret = this->InitModelBuffer(this->model_, model_buf, size, take_buf);
  MS_CHECK_TRUE_MSG(ret == RET_OK, nullptr,
                    "MindirModelLoader: Import model failed: init model buffer error with " << ret);

  // mind_ir::ModelProto model_proto;
  MS_CHECK_TRUE_MSG(this->model_->mindir_model_proto_.ParseFromArray(this->model_->buf, static_cast<int32_t>(size)),
                    nullptr, "MindirModelLoader: Import model failed, please check the correctness of the file.");

  MS_LOG(ERROR) << "model_proto: " << this->model_->mindir_model_proto_.DebugString();

  if (!this->ConvertModel(this->model_->mindir_model_proto_)) {
    MS_LOG(ERROR)
      << "MindirModelLoader: Import model failed, convert model error, please check the correctness of the file.";
    delete this->model_;
    this->model_ = nullptr;
    return nullptr;
  }

  return this->model_;
}

bool MindirModelLoader::ConvertModel(const mind_ir::ModelProto &model_proto) {
  this->model_->graph_.name_ = "";
  if (model_proto.has_model_version()) {
    this->model_->graph_.version_ = model_proto.model_version();
  }

  MS_CHECK_TRUE_MSG(
    ConvertPrimitives(model_proto), false,
    "MindirModelLoader: Import model failed, convert primitives error, please check the correctness of the file.");
  this->tensor_count_ = 0;
  this->node_count_ = 0;
  if (model_proto.has_graph()) {
    this->model_->graph_.name_ = model_proto.graph().name();
    // root graph, do not pass sub graph
    if (model_proto.functions_size() > 0) {
      MS_CHECK_TRUE_MSG(
        ConvertGraph(model_proto.graph(), nullptr, true), false,
        "MindirModelLoader: Import model failed, convert root graph error, please check the correctness of the file.");
    } else {
      // no subgraph, add graph to subgraph
      auto *sub_graph = new LiteGraph::SubGraph();
      sub_graph->name_ = model_proto.graph().name();
      MS_CHECK_TRUE_MSG(
        ConvertGraph(model_proto.graph(), sub_graph, true), false,
        "MindirModelLoader: Import model failed, convert root graph error, please check the correctness of the file.");
      this->model_->graph_.sub_graphs_.push_back(sub_graph);
    }
  }

  for (int i = 0; i < model_proto.functions_size(); i++) {
    auto sub_graph_proto = model_proto.functions(i);
    auto *sub_graph = new LiteGraph::SubGraph();
    if (sub_graph == nullptr) {
      MS_LOG(ERROR) << "MindirModelLoader: Import model failed, new sub graph failed.";
      return mindspore::lite::RET_ERROR;
    }
    // MS_CHECK_FALSE_MSG(sub_graph == nullptr, mindspore::lite::RET_ERROR,
    //                    "MindirModelLoader: Import model failed, new sub graph failed.");
    sub_graph->name_ = sub_graph_proto.name();
    MS_CHECK_TRUE_MSG(
      ConvertGraph(sub_graph_proto, sub_graph), false,
      "MindirModelLoader: Import model failed, convert sub graph error, please check the correctness of the file.");
    this->model_->graph_.sub_graphs_.push_back(sub_graph);
  }
  MS_LOG(INFO) << "MindirModelLoader: Import model successful.";
  return true;
}

bool MindirModelLoader::ConvertPrimitives(const mind_ir::ModelProto &model_proto) {
  static auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  for (int i = 0; i < model_proto.primitives_size(); i++) {
    auto primitive_proto = model_proto.primitives(i);
    auto op_type = primitive_proto.op_type();
    std::shared_ptr<mindspore::Primitive> prim;
    auto it = op_primc_fns.find(op_type);
    if (it == op_primc_fns.end()) {
      MS_LOG(WARNING) << "MindirModelLoader: Convert primitives failed, unsupported op primitive type: " << op_type;
      continue;
    }
    prim = it->second();
    MS_CHECK_TRUE_MSG(prim != nullptr, false, "MindirModelLoader: Convert primitives failed, the prim is nullptr.");
    prim->set_instance_name(op_type);
    for (int j = 0; j < primitive_proto.attribute_size(); j++) {
      auto attr_proto = primitive_proto.attribute(j);
      auto value_ptr = MindirModelUtil::MakeValueFromAttribute(attr_proto);
      MS_CHECK_TRUE_MSG(value_ptr != nullptr, false,
                        "MindirModelLoader: convert primitives failed, parse prim: "
                          << prim->ToString() << " attributes error: " << attr_proto.DebugString());
      (void)prim->AddAttr(attr_proto.name(), value_ptr);
    }
    static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
    auto op_it = operator_fns.find(op_type);
    if (op_it == operator_fns.end()) {
      MS_LOG(WARNING) << "MindirModelLoader: Convert primitives failed, unsupported op operator type: " << op_type;
      continue;
    }
    auto base_operator = op_it->second(prim);
    MS_CHECK_TRUE_MSG(this->all_operators_.count(primitive_proto.name()) <= 0, false,
                      "MindirModelLoader: There is a duplication primitive instance name: " << primitive_proto.name());
    this->all_operators_[primitive_proto.name()] = base_operator;
  }
  return true;
}

bool MindirModelLoader::ConvertGraph(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph,
                                     bool is_main_graph) {
  MS_CHECK_TRUE_MSG(
    ConvertTensors(graph_proto, sub_graph, is_main_graph), false,
    "MindirModelLoader: Convert Graph failed, convert tensors error, please check the correctness of the file.");
  MS_CHECK_TRUE_MSG(
    ConvertNodes(graph_proto, sub_graph, is_main_graph), false,
    "MindirModelLoader: Convert Graph failed, convert nodes error, please check the correctness of the file.");
  return true;
}

bool MindirModelLoader::ConvertTensors(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph,
                                       bool is_main_graph) {
  for (int i = 0; i < graph_proto.input_size(); i++) {
    const mind_ir::TensorProto &tensor_proto = graph_proto.input(i).tensor(0);
    TensorProtoWrap tensor_wrap(graph_proto.input(i).name(), tensor_proto);
    this->model_->all_mindir_tensors_.push_back(tensor_wrap);
    this->tensor_index_map_[graph_proto.input(i).name()] = this->tensor_count_;
    if (sub_graph != nullptr) {
      sub_graph->input_indices_.push_back(this->tensor_count_);
      sub_graph->tensor_indices_.push_back(this->tensor_count_);
    }
    if (is_main_graph) {
      this->model_->graph_.input_indices_.push_back(this->tensor_count_);
    }
    this->tensor_count_++;
  }
  for (int i = 0; i < graph_proto.output_size(); i++) {
    const mind_ir::TensorProto &tensor_proto = graph_proto.output(i).tensor(0);
    TensorProtoWrap tensor_wrap(graph_proto.output(i).name(), tensor_proto);
    this->model_->all_mindir_tensors_.push_back(tensor_wrap);
    this->tensor_index_map_[graph_proto.output(i).name()] = this->tensor_count_;
    if (sub_graph != nullptr) {
      sub_graph->output_indices_.push_back(this->tensor_count_);
      sub_graph->tensor_indices_.push_back(this->tensor_count_);
    }
    if (is_main_graph) {
      this->model_->graph_.output_indices_.push_back(this->tensor_count_);
    }
    this->tensor_count_++;
  }
  for (int i = 0; i < graph_proto.parameter_size(); i++) {
    const mind_ir::TensorProto &tensor_proto = graph_proto.parameter(i);
    TensorProtoWrap tensor_wrap(tensor_proto.name(), tensor_proto);
    this->model_->all_mindir_tensors_.push_back(tensor_wrap);
    this->tensor_index_map_[tensor_proto.name()] = this->tensor_count_;
    if (sub_graph != nullptr) {
      sub_graph->tensor_indices_.push_back(this->tensor_count_);
    }
    this->tensor_count_++;
  }
  return true;
}

bool MindirModelLoader::ConvertNodes(const mind_ir::GraphProto &graph_proto, LiteGraph::SubGraph *sub_graph,
                                     bool is_main_graph) {
  for (int i = 0; i < graph_proto.node_size(); i++) {
    auto node_proto = graph_proto.node(i);
    if (node_proto.op_type() == kNodeTypeConstant) {
      // Constant node, convert to tensor
      for (int j = 0; j < node_proto.attribute_size(); j++) {
        auto attribute_proto = node_proto.attribute(j);
        if (attribute_proto.type() == mind_ir::AttributeProto_AttributeType_TENSORS) {
          const mind_ir::TensorProto &tensor_proto = attribute_proto.tensors(0);
          TensorProtoWrap tensor_wrap(node_proto.name(), tensor_proto);
          this->model_->all_mindir_tensors_.push_back(tensor_wrap);
          this->tensor_index_map_[node_proto.name()] = this->tensor_count_;
          if (sub_graph != nullptr) {
            sub_graph->tensor_indices_.push_back(this->tensor_count_);
          }
          this->tensor_count_++;
        }
      }
      continue;
    }
    auto *node = new LiteGraph::Node();
    if (node == nullptr) {
      MS_LOG(ERROR) << "MindirModelLoader: Convert nodes failed, new node failed.";
      return false;
    }
    node->name_ = node_proto.name();
    node->base_operator_ = this->MakePrimitiveC(node_proto.op_type());
    auto base_operator = std::reinterpret_pointer_cast<ops::BaseOperator>(node->base_operator_);
    node->op_type_ = base_operator->GetPrim()->instance_name();

    // solve input
    for (int j = 0; j < node_proto.input_size(); j++) {
      std::string input_name = node_proto.input(j);
      auto it = this->tensor_index_map_.find(input_name);
      if (it == this->tensor_index_map_.end()) {
        MS_LOG(WARNING) << "MindirModelLoader: Convert nodes failed, cannot find input index with " << input_name;
        continue;
      }
      node->input_indices_.push_back(it->second);
    }

    // solve output
    for (int j = 0; j < node_proto.output_size(); j++) {
      std::string output_name = node_proto.output(j);
      auto it = this->tensor_index_map_.find(output_name);
      if (it == this->tensor_index_map_.end()) {
        MS_LOG(WARNING) << "MindirModelLoader: Convert nodes failed, cannot find output index with " << output_name;
        continue;
      }
      node->output_indices_.push_back(it->second);
    }

    this->model_->graph_.all_nodes_.push_back(node);
    if (sub_graph != nullptr) {
      sub_graph->node_indices_.push_back(this->node_count_);
    }
    this->node_count_++;
  }
  return true;
}

std::shared_ptr<void> MindirModelLoader::MakePrimitiveC(const std::string &node_type) {
  const std::string kOperatorTypeFlag = std::string("REF::");
  const size_t kOpTypeFlagSize = kOperatorTypeFlag.length();
  if (node_type.size() > kOpTypeFlagSize && node_type.substr(0, kOpTypeFlagSize) == kOperatorTypeFlag) {
    auto it = this->all_operators_.find(node_type.substr(kOpTypeFlagSize));
    if (it == this->all_operators_.end()) {
      MS_LOG(ERROR) << "MindirModelLoader: make primitiveC failed, can't find the primitive ref:" << node_type;
      return nullptr;
    }
    return it->second;
  }

  // node_type is not ref: pointer
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(node_type) != op_primc_fns.end()) {
    // registered primitive
    auto prim = (op_primc_fns[node_type]());
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "MindirModelLoader: Make primitiveC failed, the prim is nullptr.");
    prim->set_instance_name(node_type);
    static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
    auto op_it = operator_fns.find(node_type);
    if (op_it == operator_fns.end()) {
      MS_LOG(WARNING) << "MindirModelLoader: Make primitiveC failed, unsupported op operator type: " << node_type;
      return nullptr;
    }
    return op_it->second(prim);
  } else {
    // S_Prim_xxx or S_Prim_hyper_map[xxx] and custom node type, now not support
    MS_LOG(ERROR) << "MindirModelLoader: make primitiveC failed, not support node type: " << node_type;
    return nullptr;
  }
}

static std::shared_ptr<ModelLoader> MindirModelLoaderCreator() { return std::make_shared<MindirModelLoader>(); }

REG_MODEL_LOADER(kMindIR, MindirModelLoaderCreator);
}  // namespace mindspore::infer::mindir
