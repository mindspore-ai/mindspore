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

#include "src/expression/ops/dropout.h"
#include <vector>
#include "nnacl/fp32_grad/dropout_parameter.h"
#include "inner/model_generated.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
DropOutM::DropOutM(float ratio) {
  auto param = reinterpret_cast<DropoutParameter *>(calloc(1, sizeof(DropoutParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  param->ratio_ = ratio;
  SetOpParam(param);
  set_primitive(schema::PrimitiveType_Dropout);
  set_name(UniqueName("DropOut"));
}

std::vector<EXPR *> DropOutM::Grad(EXPR *yt) {
  auto inGrad = new (std::nothrow) DropOutGradM(this);
  if (inGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate drop grad";
    return {};
  }
  return (*inGrad)({yt, expr()});
}

int DropOutM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const DropoutParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::DropoutT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->keep_prob = param->ratio_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

DropOutGradM::DropOutGradM(DropOutM *node) {
  CloneOpParam<DropoutParameter>(node->OpParam());
  set_primitive(schema::PrimitiveType_DropoutGrad);
  set_name(kGradName + "/DropOutGrad");
}

int DropOutGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const DropoutParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::DropoutGradT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->keep_prob = param->ratio_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Dropout, ReturnNode<DropOutM>);

namespace NN {
Node *DropOut(float ratio) {
  auto node = new (std::nothrow) DropOutM(ratio);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate dropout node";
    return nullptr;
  }
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
