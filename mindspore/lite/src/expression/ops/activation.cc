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
#include "src/expression/ops/activation.h"
#include "nnacl/fp32/activation_fp32.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
ActM::ActM(schema::ActivationType type) : Node() {
  auto op_param = malloc(sizeof(ActivationParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ActivationParameter";
    return;
  }
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_Activation);
  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(opParam_.get());
  act_param->type_ = type;
  act_param->alpha_ = 0.f;
  act_param->min_val_ = 0.f;
  act_param->max_val_ = 0.f;
}

std::vector<EXPR *> ActM::Grad(EXPR *yt) {
  auto actGrad = new (std::nothrow) ActGradM(this);
  if (actGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate activation grad node";
    return {};
  }
  PushOp(actGrad);
  auto param = reinterpret_cast<ActivationParameter *>(actGrad->OpParam());
  EXPR *ag = nullptr;
  actGrad->expr()->SetSize(C2NUM);
  if ((param->type_ == schema::ActivationType_SIGMOID) || (param->type_ == schema::ActivationType_TANH)) {
    ag = (*actGrad)({output(0), yt}).front();
  } else if ((param->type_ == schema::ActivationType_HSWISH) || (param->type_ == schema::ActivationType_HSIGMOID) ||
             (param->type_ == schema::ActivationType_RELU6)) {
    ag = (*actGrad)({yt, input(0)}).front();
  } else if (param->type_ == schema::ActivationType_GELU) {
    actGrad->expr()->SetSize(C3NUM);
    ag = (*actGrad)({yt, input(0), output(0)}).front();
  } else {
    ag = (*actGrad)({yt, output(0)}).front();
  }
  std::vector<EXPR *> res = {ag};
  return res;
}

int ActM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto act_param = reinterpret_cast<const ActivationParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::ActivationT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate activation primitive";
    return RET_ERROR;
  }
  prim->activation_type = static_cast<decltype(prim->activation_type)>(act_param->type_);
  prim->alpha = act_param->alpha_;
  prim->min_val = act_param->min_val_;
  prim->max_val = act_param->max_val_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Activation, ReturnNode<ActM>);

ActGradM::ActGradM(Node *node) {
  CloneOpParam<ActivationParameter>(node->OpParam());
  set_primitive(schema::PrimitiveType_ActivationGrad);
  set_name(node->name() + "/" + kGradName + "/actGrad");
}

std::vector<EXPR *> ActGradM::Grad(EXPR *yt) { return {}; }

int ActGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto act_param = reinterpret_cast<const ActivationParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::ActivationGradT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate activation grad primitive";
    return RET_ERROR;
  }
  prim->activation_type = static_cast<decltype(prim->activation_type)>(act_param->type_);
  prim->alpha = act_param->alpha_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg regGrad(schema::PrimitiveType_ActivationGrad, ReturnNode<ActGradM>);
namespace NN {
Node *ReLU6() {
  auto r = new (std::nothrow) ActM(schema::ActivationType_RELU6);
  if (r == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate relu6";
    return nullptr;
  }
  r->set_name(Node::UniqueName("ReLU6"));
  return r;
}
Node *Sigmoid() {
  auto s = new (std::nothrow) ActM(schema::ActivationType_SIGMOID);
  if (s == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate sigmoid";
    return nullptr;
  }
  s->set_name(Node::UniqueName("Sigmoid"));
  return s;
}
Node *Relu() {
  auto r = new (std::nothrow) ActM(schema::ActivationType_RELU);
  if (r == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate relu";
    return nullptr;
  }
  r->set_name(r->UniqueName("Relu"));
  return r;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
