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

#include "src/expression/ops/biasadd.h"
#include "src/expression/ops/transpose.h"
#include "nnacl/arithmetic.h"
#include "src/expression/import.h"

namespace mindspore {
namespace lite {
BiasAddM::BiasAddM(Format data_format) {
  auto op_param = calloc(1, sizeof(ArithmeticParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ConvParameter";
    return;
  }
  auto bias_param = reinterpret_cast<ArithmeticParameter *>(op_param);
  SetOpParam(bias_param);
  set_primitive(schema::PrimitiveType_BiasAdd);
}

std::vector<EXPR *> BiasAddM::construct(const std::vector<EXPR *> &inputs) {
  auto x = Node::construct(inputs);
  AddLearn(inputs.at(C1NUM)->node());
  return x;
}

void BiasAddM::SetLearn() { AddLearn(input(C1NUM)->node()); }

std::vector<EXPR *> BiasAddM::Grad(EXPR *yt) {
  auto in = yt;
  if (yt->format() != NHWC && yt->dims().size() == C4NUM) {
    in = TransposeM::TransposeCHW2HWC(yt);
    in->node()->set_name(kGradName + "/" + name() + "/" + in->node()->name());
    PushOp(in->node());
  }
  auto grad_node = new (std::nothrow) BiasAddGradM(*this);
  if (grad_node == nullptr) {
    MS_LOG(ERROR) << "Cannon allocate Bias Grad";
    return {};
  }
  PushOp(grad_node);
  auto bias_grad = (*grad_node)({in});
  return {in, bias_grad.front()};
}
int BiasAddM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::BiasAddT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "cannot allocate prim";
    return RET_ERROR;
  }
  prim->format = static_cast<schema::Format>(KHWC);
  cnode->primitive->value.value = prim;
  return RET_OK;
}

BiasAddGradM::BiasAddGradM(const BiasAddM &bias) {
  auto op_param = calloc(1, sizeof(OpParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate op_param";
  }
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_BiasAddGrad);
}

int BiasAddGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::BiasAddGradT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_BiasAdd, ReturnNode<BiasAddM>);

namespace NN {}
}  // namespace lite
}  // namespace mindspore
