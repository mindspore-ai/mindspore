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

#include "src/expression/ops/arithmetic.h"
#include <memory>
#include "src/expression/ops/reduce.h"
#include "src/expression/ops/reshape.h"
#include "src/expression/ops_utils.h"
#include "src/expression/ops/arithmetic_self.h"
#include "src/expression/ops.h"
#include "nnacl/arithmetic.h"
#include "src/expression/import.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
// Common Arithmetic Functionality
ArithmeticM::ArithmeticM(schema::PrimitiveType type) : Node() {
  auto op_param = malloc(sizeof(ArithmeticParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ActivationParameter";
    return;
  }
  SetOpParam(op_param);
  expr()->SetSize(C2NUM);
  set_primitive(type);
}

std::vector<EXPR *> ArithmeticM::binop_grad_common(EXPR *x, EXPR *y, EXPR *dx, EXPR *dy) {
  auto shape_of_x = x->dims();
  auto shape_of_y = y->dims();
  auto reduce_dx = dx;
  auto reduce_dy = dy;
  auto rx = (BroadcastGradientArgs(shape_of_x, shape_of_y))();
  if (rx[0].size()) {
    auto reduce_sum = NN::ReduceSum(false, rx[0]);
    PushOp(reduce_sum);
    reduce_dx = (*reduce_sum)({reduce_dx}).front();
    auto reshape = NN::Reshape(shape_of_x);
    PushOp(reshape);
    reduce_dx = (*reshape)({reduce_dx}).front();
  }
  if (rx[1].size()) {
    auto reduce_sum = NN::ReduceSum(false, rx[1]);
    PushOp(reduce_sum);
    reduce_dy = (*reduce_sum)({reduce_dy}).front();
    auto reshape = NN::Reshape(shape_of_y);
    PushOp(reshape);
    reduce_dy = (*reshape)({reduce_dy}).front();
  }
  std::vector<EXPR *> out = {reduce_dx, reduce_dy};
  return out;
}

// Add Op
AddM::AddM(int dummy) : ArithmeticM(schema::PrimitiveType_AddFusion) { set_name(UniqueName("Add")); }

std::vector<EXPR *> AddM::Grad(EXPR *yt) { return binop_grad_common(input(0), input(1), yt, yt); }

int AddM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::AddFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate prim";
    return RET_ERROR;
  }
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg AddReg(schema::PrimitiveType_AddFusion, ReturnNode<AddM>);

// Div op
DivM::DivM(int dummy) : ArithmeticM(schema::PrimitiveType_RealDiv) { set_name(UniqueName("RealDiv")); }
std::vector<EXPR *> DivM::Grad(EXPR *yt) {
  auto x = input(0);
  auto y = input(1);
  auto o = output(0);
  auto div_op = NN::Div();
  if (div_op == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate div_op";
    return {};
  }
  PushOp(div_op);
  auto neg_op = NN::Neg();
  if (neg_op == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate neg_op";
    return {};
  }
  PushOp(neg_op);
  auto mul_op = NN::Mul();
  if (mul_op == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate mul_op";
    return {};
  }
  PushOp(mul_op);
  auto bc_x = (*div_op)({yt, y}).front();
  auto bc_y = (*neg_op)((*mul_op)({bc_x, o})).front();
  return binop_grad_common(x, y, bc_x, bc_y);
}
int DivM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::RealDivT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}
static ImportReg DivReg(schema::PrimitiveType_DivFusion, ReturnNode<DivM>);

// Mul op
MulM::MulM(int dummy) : ArithmeticM(schema::PrimitiveType_MulFusion) { set_name(UniqueName("Mul")); }

std::vector<EXPR *> MulM::Grad(EXPR *yt) {
  auto mul_dx = NN::Mul();
  if (mul_dx == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate mul dx";
    return {};
  }
  PushOp(mul_dx);
  auto mul_dy = NN::Mul();
  if (mul_dy == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate mul_dy";
    return {};
  }
  PushOp(mul_dy);
  auto x = input(0);
  auto y = input(1);
  auto bc_dx = (*mul_dx)({y, yt}).front();
  auto bc_dy = (*mul_dy)({x, yt}).front();
  return binop_grad_common(x, y, bc_dx, bc_dy);
}

int MulM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::MulFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate prim";
    return RET_ERROR;
  }
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  cnode->primitive->value.value = prim;
  return RET_OK;
}
static ImportReg MulReg(schema::PrimitiveType_MulFusion, ReturnNode<MulM>);

// Sub op
SubM::SubM(int dummy) : ArithmeticM(schema::PrimitiveType_SubFusion) { set_name(UniqueName("Sub")); }

std::vector<EXPR *> SubM::Grad(EXPR *yt) {
  auto x = input(0);
  auto y = input(1);
  auto neg = NN::Neg();
  if (neg == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate neg";
    return {};
  }
  PushOp(neg);
  auto neg_grad = (*neg)({yt}).front();
  return binop_grad_common(x, y, yt, neg_grad);
}
int SubM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::SubFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate prim";
    return RET_ERROR;
  }
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  cnode->primitive->value.value = prim;
  return RET_OK;
}
static ImportReg SubReg(schema::PrimitiveType_SubFusion, ReturnNode<SubM>);

namespace NN {
Node *Add() {
  auto a = new (std::nothrow) AddM(0);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate a";
    return nullptr;
  }
  return a;
}
Node *Sub() {
  auto a = new (std::nothrow) SubM(0);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate a";
    return nullptr;
  }
  return a;
}

Node *Mul() {
  auto a = new (std::nothrow) MulM(0);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate a";
    return nullptr;
  }
  return a;
}
Node *Div() {
  auto a = new (std::nothrow) DivM(0);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate a";
    return nullptr;
  }
  return a;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
