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

#include "src/expression/ops/flatten.h"
#include <vector>
#include "inner/model_generated.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"

#include "src/litert/cxx_api/expression/node_impl.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
FlattenM::FlattenM(int dummy) {
  auto param = reinterpret_cast<OpParameter *>(calloc(C1NUM, sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  SetOpParam(param);
  set_primitive(schema::PrimitiveType_Flatten);
  set_name(UniqueName("Flatten"));
}

std::vector<EXPR *> FlattenM::construct(const std::vector<EXPR *> &inputs) {
  auto in = inputs;
  auto y = Node::construct(in);
  return y;
}

std::vector<EXPR *> FlattenM::Grad(EXPR *yt) {
  auto shape_of_x = input(0)->dims();
  auto reshape = NN::Reshape(shape_of_x);
  PushOp(reshape);
  return (*reshape)({yt});
}

int FlattenM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::DropoutT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Flatten, ReturnNode<FlattenM>);

namespace NN {
Node *Flatten() {
  auto node = new (std::nothrow) FlattenM(0);
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
