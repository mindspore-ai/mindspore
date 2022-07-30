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

#include "src/expression/ops/softmaxCE.h"
#include "include/api/net.h"
#include "src/litert/cxx_api/expression/node_impl.h"
#include "src/expression/ops/reduce.h"
namespace mindspore {
namespace NN {
Node *SoftmaxCrossEntropy(const SoftMaxCrossEntropyCfg &cfg) {
  auto lite_node = lite::NN::SoftmaxCrossEntropy(cfg);
  return NodeImpl::Connect(lite_node);
}
}  // namespace NN

namespace lite {
SoftmaxCrossEntropyM::SoftmaxCrossEntropyM() {
  auto param = calloc(1, sizeof(OpParameter));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  expr()->SetSize(C2NUM);
  SetOpParam(param);
  set_name("SoftmaxCrossEntropy");
  set_primitive(schema::PrimitiveType_SoftmaxCrossEntropyWithLogits);
  EXPR e(this);
  e.SetSize(0);
  expr_.emplace_back(e);
}

Node *SoftmaxCrossEntropyM::GetReductionNode(const std::string &mode, const std::vector<int> &axis) {
  if (mode == "mean") {
    return NN::ReduceMean(false, axis);
  } else if (mode == "sum") {
    return NN::ReduceSum(false, axis);
  } else {
    return nullptr;
  }
}

SoftmaxCrossEntropyM::SoftmaxCrossEntropyM(const SoftMaxCrossEntropyCfg &cfg) : SoftmaxCrossEntropyM() {
  std::vector<int> axis = {0};
  reduce_ = GetReductionNode(cfg.reduction, axis);
  if (reduce_ != nullptr) {
    PushOp(reduce_);
  }
}

std::vector<EXPR *> SoftmaxCrossEntropyM::construct(const std::vector<EXPR *> &inputs) {
  auto y = Node::construct(inputs);
  if (reduce_ != nullptr) {
    y = (*reduce_)({y.front()});
  }
  return y;
}

std::vector<EXPR *> SoftmaxCrossEntropyM::Grad(EXPR *expr) { return {this->expr(1)}; }

int SoftmaxCrossEntropyM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::SoftmaxCrossEntropyWithLogitsT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

namespace NN {
Node *SoftmaxCrossEntropy(const SoftMaxCrossEntropyCfg &cfg) {
  auto s = new (std::nothrow) SoftmaxCrossEntropyM(cfg);
  if (s == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate softmax node";
  }
  return s;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
