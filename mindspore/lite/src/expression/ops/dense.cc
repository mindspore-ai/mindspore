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

#include "src/expression/ops/dense.h"
#include <memory>
#include "include/api/cfg.h"
#include "src/expression/ops/biasadd.h"
#include "src/expression/ops/depend.h"
#include "src/expression/ops.h"
#include "nnacl/matmul_parameter.h"
#include "src/expression/import.h"
#include "inner/model_generated.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
DenseM::DenseM(const DenseConfig &cfg) : Node() {
  auto op_param = calloc(1, sizeof(MatMulParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate MatMulParameter";
    return;
  }
  set_name(UniqueName("Dense"));
  SetOpParam(op_param);
  expr()->SetSize(C2NUM);
  set_primitive(schema::PrimitiveType_MatMulFusion);
  auto param = reinterpret_cast<MatMulParameter *>(opParam_.get());
  param->row_ = cfg.out_channels_;
  param->col_ = cfg.in_channels_;
  param->a_transpose_ = false;
  param->b_transpose_ = true;
  std::vector<int> dims = {param->row_, param->col_};
  auto w = Node::CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::Mode::NORMAL, "weights");
  expr()->set_params(C1NUM, w);
  if (cfg.has_bias_) {
    wbias_ = CreateWeights({cfg.out_channels_}, kNumberTypeFloat32, KHWC, Param::Mode::ZEROS, "bias_weights");
    bias_ = new (std::nothrow) BiasAddM(KHWC);
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "Cannot allocate bias";
      return;
    }
    bias_->update_name(name());
    AddLearn(wbias_->node());
    PushOp(bias_);
  }
  SetLearn();
}

std::vector<EXPR *> DenseM::construct(const std::vector<EXPR *> &inputs) {
  auto x = Node::construct(inputs);
  if (bias_ != nullptr) {
    x = (*bias_)({x.front(), wbias_});
  }
  return x;
}

std::vector<EXPR *> DenseM::Grad(EXPR *yt) {
  auto src_param = reinterpret_cast<MatMulParameter *>(opParam_.get());
  bool ta = src_param->a_transpose_;
  bool tb = src_param->b_transpose_;

  // dx grad op
  auto dxGrad = new (std::nothrow) DenseM();
  if (dxGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate dxGrad ";
    return {};
  }
  PushOp(dxGrad);
  dxGrad->CloneOpParam<MatMulParameter>(opParam_);
  dxGrad->set_primitive(schema::PrimitiveType_MatMulFusion);
  auto dxGradParam = reinterpret_cast<MatMulParameter *>(dxGrad->OpParam());
  dxGradParam->a_transpose_ = (ta && tb);
  dxGradParam->b_transpose_ = (ta || !tb);
  dxGrad->set_name(name() + kGradName + "/dxGrad");
  EXPR *dx = nullptr;
  if (ta) {
    dx = (*dxGrad)({input(1), yt}).front();
  } else {
    dx = (*dxGrad)({yt, input(1)}).front();
  }
  // Control execution flow
  auto depend = NN::Depend();
  if (depend == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate depend ";
    return {};
  }
  PushOp(depend);
  auto de = (*depend)({dxGrad->expr()}).front();

  // dw grad op
  auto dwGrad = new (std::nothrow) DenseM();
  if (dwGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate dwGrad ";
    return {};
  }
  PushOp(dwGrad);
  dwGrad->CloneOpParam<MatMulParameter>(opParam_);
  dwGrad->set_primitive(schema::PrimitiveType_MatMulFusion);
  auto dwGradParam = reinterpret_cast<MatMulParameter *>(dwGrad->OpParam());
  dwGradParam->a_transpose_ = (!ta || tb);
  dwGradParam->b_transpose_ = ta && tb;
  dwGrad->set_name(name() + kGradName + "/dwGrad");
  EXPR *dw = nullptr;
  if (tb) {
    dw = (*dwGrad)({yt, input(0), de}).front();
  } else {
    dw = (*dwGrad)({input(0), yt, de}).front();
  }
  return {dx, dw};
}
int DenseM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto dense_param = reinterpret_cast<const MatMulParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::MatMulFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->transpose_a = dense_param->a_transpose_;
  prim->transpose_b = dense_param->b_transpose_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

void DenseM::SetLearn() { AddLearn(input(C1NUM)->node()); }

static ImportReg reg(schema::PrimitiveType_MatMulFusion, ReturnNode<DenseM>);

namespace NN {
Node *Dense(const DenseConfig &cfg) {
  auto l = new (std::nothrow) DenseM(cfg);
  if (l == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate Dense object";
  }
  return l;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
