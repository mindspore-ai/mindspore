/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ut/tools/converter/registry/parser/model_parser_test.h"
#include <map>
#include <vector>
#include "include/errorcode.h"
#include "include/registry/model_parser_registry.h"

namespace mindspore {
api::FuncGraphPtr ModelParserTest::Parse(const converter::ConverterParameters &flag) {
  // construct funcgraph
  res_graph_ = api::FuncGraph::Create();
  auto ret = InitOriginModelStructure();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "obtain origin model structure failed.";
    return nullptr;
  }
  ret = BuildGraphInputs();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "build graph inputs failed.";
    return nullptr;
  }
  ret = BuildGraphNodes();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "build graph nodes failed.";
    return nullptr;
  }
  ret = BuildGraphOutputs();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "build graph outputs failed.";
    return nullptr;
  }
  return res_graph_;
}

int ModelParserTest::InitOriginModelStructure() {
  model_structure_ = {"add_0", "add_1"};
  model_layers_info_ = {{"input", {"graph_input0"}},
                        {"add_0", {"graph_input0", "const_0"}},
                        {"add_1", {"add_0", "const_1"}},
                        {"output", {"add_1"}}};
  return lite::RET_OK;
}

int ModelParserTest::BuildGraphInputs() {
  if (model_layers_info_.find("input") == model_layers_info_.end()) {
    MS_LOG(ERROR) << "model is invalid";
    return lite::RET_ERROR;
  }
  auto inputs = model_layers_info_["input"];
  for (auto &input : inputs) {
    auto parameter = res_graph_->add_parameter();
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "build parameter node failed.";
      return lite::RET_ERROR;
    }
    ShapeVector shape{10, 10};
    auto tensor_info = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeFloat32, shape);
    if (tensor_info == nullptr) {
      return lite::RET_ERROR;
    }
    parameter->set_name(input);
    parameter->set_abstract(tensor_info->ToAbstract());
    nodes_.insert(std::make_pair(input, parameter));
  }
  return lite::RET_OK;
}

int ModelParserTest::BuildGraphNodes() {
  if (model_structure_.empty()) {
    MS_LOG(ERROR) << "model is invalid.";
    return lite::RET_ERROR;
  }
  for (auto &node_name : model_structure_) {
    if (model_layers_info_.find(node_name) == model_layers_info_.end()) {
      MS_LOG(ERROR) << "model is invalid.";
      return lite::RET_ERROR;
    }
    auto node_inputs = model_layers_info_[node_name];
    if (node_inputs.empty()) {
      MS_LOG(ERROR) << "model is invalid.";
      return lite::RET_ERROR;
    }
    auto type = node_name.substr(0, node_name.find_last_of("_"));
    auto node_parser = NodeParserTestRegistry::GetInstance()->GetNodeParser(type);
    if (node_parser == nullptr) {
      MS_LOG(ERROR) << "cannot find current op parser.";
      return lite::RET_ERROR;
    }
    auto primc = node_parser->Parse();
    if (primc == nullptr) {
      MS_LOG(ERROR) << "node parser failed.";
      return lite::RET_ERROR;
    }
    std::vector<AnfNodePtr> anf_inputs;
    for (auto &input : node_inputs) {
      if (nodes_.find(input) != nodes_.end()) {
        anf_inputs.push_back(nodes_[input]);
      } else {
        auto parameter = res_graph_->add_parameter();
        if (parameter == nullptr) {
          MS_LOG(ERROR) << "build parameter node failed.";
          return lite::RET_ERROR;
        }
        ShapeVector shape{10, 10};
        auto tensor_info = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeFloat32, shape);
        auto size = tensor_info->Size();
        memset_s(tensor_info->data_c(), size, 0, size);
        parameter->set_abstract(tensor_info->ToAbstract());
        parameter->set_default_param(tensor_info);
        parameter->set_name(input);
        anf_inputs.push_back(parameter);
        nodes_.insert(std::make_pair(input, parameter));
      }
    }
    auto cnode = res_graph_->NewCNode(std::shared_ptr<ops::PrimitiveC>(primc), anf_inputs);
    cnode->set_fullname_with_scope(node_name);
    auto tensor_info = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeFloat32, ShapeVector{});
    cnode->set_abstract(tensor_info->ToAbstract());
    nodes_.insert(std::make_pair(node_name, cnode));
  }
  return lite::RET_OK;
}

int ModelParserTest::BuildGraphOutputs() {
  if (model_layers_info_.find("output") == model_layers_info_.end()) {
    MS_LOG(ERROR) << "model is invalid.";
    return lite::RET_ERROR;
  }
  auto outputs = model_layers_info_["output"];
  if (outputs.empty()) {
    MS_LOG(ERROR) << "odel is invalid.";
    return lite::RET_ERROR;
  }
  if (outputs.size() > 1) {
    // need generate a MakeTuple to package outputs.
  } else {
    if (nodes_.find(outputs[0]) == nodes_.end()) {
      return lite::RET_ERROR;
    }
    auto return_prim = std::make_shared<Primitive>("Return");
    auto return_cnode = res_graph_->NewCNode(return_prim, {nodes_[outputs[0]]});
    return_cnode->set_fullname_with_scope("Return");
    res_graph_->set_return(return_cnode);
  }
  return lite::RET_OK;
}

converter::ModelParser *TestModelParserCreator() {
  auto *model_parser = new (std::nothrow) ModelParserTest();
  if (model_parser == nullptr) {
    MS_LOG(ERROR) << "new model parser failed";
    return nullptr;
  }
  return model_parser;
}
}  // namespace mindspore
