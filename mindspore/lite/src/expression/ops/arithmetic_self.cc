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

#include "src/expression/ops/arithmetic_self.h"
#include <memory>
#include "src/expression/ops_utils.h"
#include "src/expression/ops.h"
#include "nnacl/arithmetic_self_parameter.h"
#include "src/expression/import.h"

namespace mindspore {
namespace lite {
// Common Arithmetic Self Functionality
ArithmeticSelfM::ArithmeticSelfM(schema::PrimitiveType type) : Node() {
  auto op_param = malloc(sizeof(ArithmeticSelfParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ArithmeticSelfParameter";
    return;
  }
  memset(op_param, 0, sizeof(ArithmeticSelfParameter));
  SetOpParam(op_param);
  expr()->SetSize(C1NUM);
  set_primitive(type);
}

// NEG OP
NegM::NegM(int dummy) : ArithmeticSelfM(schema::PrimitiveType_NegGrad) { set_name(UniqueName("Neg")); }

std::vector<EXPR *> NegM::Grad(EXPR *yt) {
  auto grad_neg = new (std::nothrow) NegM(0);
  if (grad_neg == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate neg gradient";
    return {};
  }
  return (*grad_neg)({yt});
}

int NegM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::NegT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

namespace NN {
Node *Neg() {
  auto a = new (std::nothrow) NegM(0);
  if (a == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate neg node";
    return nullptr;
  }
  return a;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
