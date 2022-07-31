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

#include "src/expression/ops/batchnorm.h"
#include <memory>
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
BatchNorm2dM::BatchNorm2dM(int outp, float momentum, float epsilon) {
  constexpr int bn_inputs = 5;
  constexpr int bn_outputs = 5;

  auto op_param = calloc(1, sizeof(BatchNormParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate BatchNormParameter";
    return;
  }
  expr()->SetSize(bn_inputs);
  set_name(UniqueName("BatchNorm2D"));
  auto bn_param = reinterpret_cast<BatchNormParameter *>(op_param);
  bn_param->channel_ = outp;
  bn_param->momentum_ = momentum;
  bn_param->epsilon_ = epsilon;
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_FusedBatchNorm);
  std::vector<int> dims = {outp};
  auto scale = Node::CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::Mode::ONES, "scale");
  expr()->set_params(C1NUM, scale);
  auto offset = Node::CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::Mode::ZEROS, "offset");
  expr()->set_params(C2NUM, offset);
  auto mean = Node::CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::Mode::ZEROS, "mean");
  expr()->set_params(C3NUM, mean);
  auto var = Node::CreateWeights(dims, kNumberTypeFloat32, KHWC, Param::Mode::ONES, "var");
  expr()->set_params(C4NUM, var);
  SetOutputs(bn_outputs);
  SetLearn();
}

BatchNorm2dGradM::BatchNorm2dGradM(BatchNorm2dM *bn_node) : Node() {
  auto op_param = calloc(1, sizeof(BNGradParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate BNGradParameter";
    return;
  }
  expr()->SetSize(C6NUM);
  set_name(bn_node->name() + "/" + kGradName + "/bnGrad");
  auto bn_grad_param = reinterpret_cast<BNGradParameter *>(op_param);
  auto bn_param = reinterpret_cast<BatchNormParameter *>(bn_node->OpParam());
  bn_param->is_training_ = true;
  bn_grad_param->epsilon_ = bn_param->epsilon_;
  bn_grad_param->is_training_ = true;
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_BatchNormGrad);
  EXPR e(this);
  e.SetSize(0);
  // Dgamma
  expr_.emplace_back(e);
  // Doffset
  expr_.emplace_back(e);
}

std::vector<EXPR *> BatchNorm2dM::Grad(EXPR *yt) {
  auto bn_grad_node = new (std::nothrow) BatchNorm2dGradM(this);
  if (bn_grad_node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate batchnorm grad";
    return {};
  }
  PushOp(bn_grad_node);
  auto bn_grad = (*bn_grad_node)({yt, input(0), output(1), output(3), output(4), output(2)});
  return bn_grad;
}

int BatchNorm2dM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto bn_param = reinterpret_cast<const BatchNormParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::FusedBatchNormT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->epsilon = bn_param->epsilon_;
  prim->momentum = bn_param->momentum_;
  prim->mode = (bn_param->is_training_ == false) ? 0 : 1;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

void BatchNorm2dM::SetLearn() {
  AddLearn(input(C1NUM)->node());
  AddLearn(input(C2NUM)->node());
}

int BatchNorm2dGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const BNGradParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::BatchNormGradT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->epsilon = param->epsilon_;
  prim->is_training = param->is_training_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_FusedBatchNorm, ReturnNode<BatchNorm2dM>);
namespace NN {
Node *BatchNorm2D(int outp, float momentum, float epsilon) {
  auto node = new (std::nothrow) BatchNorm2dM(outp, momentum, epsilon);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate node";
    return nullptr;
  }
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
