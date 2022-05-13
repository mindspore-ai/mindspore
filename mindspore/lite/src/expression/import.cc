/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include "src/expression/import.h"
#include "common/ops/populate/populate_register.h"
#include "src/expression/ops.h"
#include "src/expression/ops/activation.h"
#include "src/expression/ops/batchnorm.h"
#include "src/expression/ops/biasadd.h"
#include "src/expression/ops/conv.h"
#include "src/expression/ops/dense.h"
#include "src/expression/ops/pooling.h"
#include "src/expression/ops/reshape.h"
#include "src/expression/ops/transpose.h"

namespace mindspore {
namespace lite {
std::unordered_map<mindspore::schema::PrimitiveType, import_func> ImportReg::import_map_;

import_func ImportReg::GetImportFunc(mindspore::schema::PrimitiveType type) {
  auto f = import_map_.find(type);
  if (f == import_map_.end()) {
    return nullptr;
  }
  return f->second;
}

OpParameter *Import::GetAttr(const schema::Primitive *prim) {
  auto parameter_gen = PopulateRegistry::GetInstance()->GetParameterCreator(prim->value_type(), SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
    return nullptr;
  }
  auto parameter = parameter_gen(prim);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }
  return parameter;
}

std::unique_ptr<Node> Import::CreateNode(const schema::CNode *cnode) {
  auto param = GetAttr(cnode->primitive());
  auto type = cnode->primitive()->value_type();
  auto fn = ImportReg::GetImportFunc(type);
  if (fn == nullptr) {
    MS_LOG(ERROR) << "Cannot find importer for " << schema::EnumNamePrimitiveType(type);
    return nullptr;
  }
  auto node = fn();
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate node" << cnode->name()->str();
    return nullptr;
  }
  node->SetOpParam(param);
  node->set_name(cnode->name()->str());
  node->set_primitive(type);
  return std::unique_ptr<Node>(node);
}

Net *Import::ImportMs(std::string file_name) {
  std::ifstream infile;
  infile.open(file_name, std::ios::binary | std::ios::in);
  if (!infile.good()) {
    MS_LOG(ERROR) << "cannot read " << file_name << std::endl;
    return nullptr;
  }
  infile.seekg(0, std::ios::end);
  int length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  auto data_ptr = std::make_unique<int8_t[]>(length);
  auto *data = data_ptr.get();
  infile.read(reinterpret_cast<char *>(data), length);
  infile.close();
  flatbuffers::Verifier verifier = flatbuffers::Verifier(reinterpret_cast<const uint8_t *>(data), length);
  bool res = schema::VerifyMetaGraphBuffer(verifier);
  if (res != true) {
    MS_LOG(ERROR) << "fault file: " << file_name << "(" << length << ")\n";
    return nullptr;
  } else {
    MS_LOG(INFO) << "verify pass file: " << file_name << "(" << length << ")\n";
  }
  buffer_ = data_ptr.get();
  auto metaGraph = schema::GetMetaGraph(data_ptr.release());
  return ImportMs(metaGraph);
}

Net *Import::ImportMs(const schema::MetaGraph *metaGraph) {
  if (metaGraph == nullptr) {
    MS_LOG(ERROR) << "null input";
    return nullptr;
  }
  std::string NetName = "Network";
  if (metaGraph->name() != nullptr) NetName = metaGraph->name()->str();
  auto net = std::make_unique<Net>(NetName);
  std::unordered_map<int, EXPR *> outputs;
  // save inputs
  for (size_t i = 0; i < metaGraph->inputIndex()->size(); i++) {
    auto tensor_id = metaGraph->inputIndex()->Get(i);
    const schema::Tensor *tensor = metaGraph->allTensors()->Get(tensor_id);
    auto input = new (std::nothrow) InputM(tensor);
    if (input == nullptr) {
      MS_LOG(ERROR) << "Cannot allocate input";
      return nullptr;
    }
    auto e = input->expr();
    outputs[tensor_id] = e;
    net->PushInput(e);
  }
  for (size_t i = 0; i < metaGraph->nodes()->size(); i++) {
    auto Cnode = metaGraph->nodes()->Get(i);
    std::vector<EXPR *> param_tensors;
    for (size_t j = 0; j < Cnode->inputIndex()->size(); j++) {
      int tensor_id = Cnode->inputIndex()->Get(j);
      const schema::Tensor *tensor = metaGraph->allTensors()->Get(tensor_id);
      auto iter = outputs.find(tensor_id);
      if (iter == outputs.end()) {
        // create value node if not exist
        if (tensor->nodeType() != NodeType::NodeType_CNode) {
          auto valnode = new (std::nothrow) InputM(tensor);
          if (valnode == nullptr) {
            MS_LOG(ERROR) << "Cannot allocate valnode";
            return nullptr;
          }
          outputs[tensor_id] = valnode->expr();
          param_tensors.push_back(valnode->expr());
          net->PushOp(valnode);
        } else {
          MS_LOG(ERROR) << "did not found input tensor " << tensor_id;
          return nullptr;
        }
      } else {
        param_tensors.push_back(iter->second);
      }
    }
    // create expression from node //
    auto node = CreateNode(Cnode);
    if (node != nullptr) {
      node->SetOutputs(Cnode->outputIndex()->size());
      std::vector<EXPR *> e = (*node)(param_tensors);
      for (size_t j = 0; j < Cnode->outputIndex()->size(); j++) {
        int tensor_id = Cnode->outputIndex()->Get(j);
        outputs[tensor_id] = e.at(j);
      }
    } else {
      MS_LOG(ERROR) << "failed to create node " << Cnode->name();
      return nullptr;
    }
    auto node_ptr = node.release();
    net->PushOp(node_ptr);
    node_ptr->SetLearn();
  }
  for (size_t i = 0; i < metaGraph->outputIndex()->size(); i++) {
    auto tensor_id = metaGraph->outputIndex()->Get(i);
    auto iter = outputs.find(tensor_id);
    if (iter == outputs.end()) {
      MS_LOG(ERROR) << "could not find source for tensor " << tensor_id;
      return nullptr;
    } else {
      net->PushOutput(iter->second);
    }
  }
  return net.release();
}
}  // namespace lite
}  // namespace mindspore
