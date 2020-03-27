/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/node.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "schema/inner/ms_generated.h"
#include "common/mslog.h"
#include "common/op_utils.h"
#include "include/errorcode.h"
#include "src/op_factory.h"

namespace mindspore {
namespace predict {
Node::Node(const NodeDef *nodeDef)
    : id(std::string(nodeDef->opDef()->name()->c_str())), type(GetOpTypeName(*nodeDef)) {}

Node::~Node() {
  if (op != nullptr) {
    delete op;
  }
}

NODE_ID Node::ID() { return id; }

std::string Node::Type() { return type; }

void Node::SetTensors(const NodeDef &nodeDef, const std::vector<Tensor *> &allTensors) {
  if (nodeDef.opDef() == nullptr) {
    MS_LOGE("nodeDef is null");
    return;
  }

  auto inputIndex = nodeDef.opDef()->inputIndex();
  MS_ASSERT(inputIndex != nullptr);
  inputs.resize(inputIndex->size());
  std::transform(inputIndex->begin(), inputIndex->end(), inputs.begin(), [allTensors](int i) { return allTensors[i]; });

  auto outputIndex = nodeDef.opDef()->outputIndex();
  MS_ASSERT(outputIndex != nullptr);
  outputs.resize(outputIndex->size());
  std::transform(outputIndex->begin(), outputIndex->end(), outputs.begin(),
                 [allTensors](int i) { return allTensors[i]; });
}

void Node::SetDepends(const std::unordered_set<NODE_ID> &deps) { depends = deps; }

std::unordered_set<NODE_ID> Node::GetDepends() { return depends; }

void Node::AddInEdge(Node *node) {
  if (node == nullptr) {
    MS_LOGE("node is null");
    return;
  }
  inEdges.insert(node);
}

void Node::AddOutEdge(Node *node) {
  if (node == nullptr) {
    MS_LOGE("node is null");
    return;
  }
  outEdges.insert(node);
}

std::unordered_set<Node *> &Node::GetAllInEdges() { return inEdges; }

std::unordered_set<Node *> &Node::GetAllOutEdges() { return outEdges; }

std::vector<Tensor *> &Node::GetOutputTensors() { return outputs; }
std::vector<Tensor *> &Node::GetInputTensors() { return inputs; }

int Node::InitOp(const OpDef &opDef, const Context &ctx) {
  OpDesc dst;
  dst.type = GetOpType(opDef);
  dst.arch = X86_FP32;
  MS_ASSERT(OpFactory::GetInstance() != nullptr);
  op = OpFactory::GetInstance()->GetOp(inputs, outputs, opDef, ctx, dst);
  if (op == nullptr) {
    MS_LOGE("Can't find opName: %s, type: %s ", id.c_str(), type.c_str());
    return RET_ERROR;
  }
  return RET_OK;
}

int Node::Run(const Context &ctx) {
  MS_LOGD("%s run start", id.c_str());
  auto ret = MallocOutput(ctx);
  if (ret != RET_OK) {
    MS_LOGE("MallocOutput failed: %d", ret);
    return ret;
  }
  if (op == nullptr) {
    MS_LOGE("op is nullptr.");
    return RET_ERROR;
  }
  ret = op->Execute(inputs, outputs);
  if (ret != RET_OK) {
    return ret;
  }
  FreeInput();
  return RET_OK;
}

int Node::MallocOutput(const Context &ctx) {
  size_t refCount = outEdges.size();
  for (auto tensor : outputs) {
    if (tensor == nullptr) {
      MS_LOGE("tensor in outputs is nullptr");
      return RET_ERROR;
    }
    auto ret = tensor->MallocData(ctx.allocator, refCount);
    if (ret != RET_OK) {
      return ret;
    }
  }
  return RET_OK;
}

void Node::FreeInput() {
  for (auto tensor : inputs) {
    if (tensor == nullptr) {
      MS_LOGW("tensor in inputs is nullptr");
      return;
    }
    if (tensor->RefCount() != MSConst_WEIGHT_REFCOUNT) {
      tensor->FreeData();
    }
  }
}
}  // namespace predict
}  // namespace mindspore
