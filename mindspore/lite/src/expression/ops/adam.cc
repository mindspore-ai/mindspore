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

#include "src/expression/ops/adam.h"
#include <memory>
#include <set>
#include <utility>
#include "src/expression/ops.h"
#include "src/expression/ops/assign.h"
#include "src/expression/ops/arithmetic.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "include/api/net.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace NN {
Node *Adam(std::shared_ptr<NodeSet> learn, const AdamConfig &cfg) {
  auto lite_node = lite::NN::Adam(std::move(learn->set_), cfg);
  return NodeImpl::Connect(lite_node);
}
}  // namespace NN

namespace lite {
std::vector<EXPR *> AdamM::Clone(EXPR *grad, EXPR *weight) {
  auto adam = new (std::nothrow) AdamM();
  if (adam == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate adam";
    return {};
  }
  adam->set_name("optimizer-Adam");
  adam->CloneOpParam<AdamParameter>(OpParam());
  adam->update_name(weight->node()->name());
  adam->set_primitive(primitive());
  adam->expr()->SetSize(C10NUM);
  // setup weight and momentum
  adam->expr()->set_params(C0NUM, weight);
  auto dims = grad->dims();
  auto m = adam->CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::ZEROS, "m");
  adam->expr()->set_params(C1NUM, m);
  auto v = adam->CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::ZEROS, "v");
  adam->expr()->set_params(C2NUM, v);
  // copy parameters
  for (int i = C3NUM; i < C9NUM; i++) {
    adam->expr()->set_params(i, this->input(i));
  }
  adam->expr()->set_params(C9NUM, grad);
  return (*adam)(adam->inputs());
}

AdamM::AdamM(std::set<Node *> &&learn, const AdamConfig &cfg) {
  auto op_param = reinterpret_cast<AdamParameter *>(malloc(sizeof(AdamParameter)));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ActivationParameter";
    return;
  }
  AssignLearn(std::move(learn));
  memset(op_param, 0, sizeof(AdamParameter));
  op_param->use_nesterov_ = cfg.use_nesterov_;
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_Adam);
  set_name("optimizer-Adam");
  // Adam Network
  expr()->SetSize(C10NUM);
  auto assign1 = new (std::nothrow) AssignM(0);
  if (assign1 == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate assign";
    return;
  }
  PushOp(assign1);
  auto assign2 = new (std::nothrow) AssignM(0);
  if (assign2 == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate assign";
    return;
  }
  PushOp(assign2);
  auto mul1 = NN::Mul();
  if (mul1 == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate mul";
    return;
  }
  PushOp(mul1);
  auto mul2 = NN::Mul();
  if (mul2 == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate mul";
    return;
  }
  PushOp(mul2);
  auto tmp = 1.0f;
  mul1->CreateConstTensor(C0NUM, {1}, kNumberTypeFloat32, KHWC, "beta1-power", &tmp);
  mul1->CreateConstTensor(C1NUM, {1}, kNumberTypeFloat32, KHWC, "beta1-data", &cfg.beta1_);
  auto o1 = (*mul1)({});
  assign1_ = (*assign1)({mul1->input(0), o1.front()}).front();
  mul2->CreateConstTensor(C0NUM, {1}, kNumberTypeFloat32, KHWC, "beta2-power", &tmp);
  mul2->CreateConstTensor(C1NUM, {1}, kNumberTypeFloat32, KHWC, "beta2-data", &cfg.beta2_);
  auto o2 = (*mul2)({});
  assign2_ = (*assign2)({mul2->input(0), o2.front()}).front();
  expr()->set_params(C3NUM, o1.front());
  expr()->set_params(C4NUM, o2.front());
  CreateConstTensor(C5NUM, {1}, kNumberTypeFloat32, KHWC, "learning-rate", &cfg.learning_rate_);
  CreateConstTensor(C6NUM, {1}, kNumberTypeFloat32, KHWC, "beta1", &cfg.beta1_);
  CreateConstTensor(C7NUM, {1}, kNumberTypeFloat32, KHWC, "beta2", &cfg.beta2_);
  CreateConstTensor(C8NUM, {1}, kNumberTypeFloat32, KHWC, "epsilon", &cfg.eps_);
}

int AdamM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const AdamParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::AdamT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate " << cnode->name;
    return RET_ERROR;
  }
  prim->use_nesterov = param->use_nesterov_;
  prim->use_locking = false;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

namespace NN {
Node *Adam(std::set<Node *> &&learn, const AdamConfig &cfg) {
  auto a = new (std::nothrow) AdamM(std::move(learn), cfg);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate adam";
    return nullptr;
  }
  return a;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
