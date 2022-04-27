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

#include "src/expression/ops/reshape.h"
#include "src/expression/ops.h"
#include "nnacl/reshape_parameter.h"
#include "src/expression/import.h"

namespace mindspore {
namespace lite {
ReshapeM::ReshapeM(const std::vector<int> &shape) : Node() {
  auto op_param = calloc(1, sizeof(ReshapeParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ReshapeParameter";
    return;
  }
  set_name(UniqueName("Reshape"));
  expr()->SetSize(C2NUM);
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_Reshape);

  ReshapeParameter *reshape_param = reinterpret_cast<ReshapeParameter *>(opParam_.get());
  reshape_param->shape_dim_ = shape.size();
  for (int i = 0; i < reshape_param->shape_dim_; i++) {
    reshape_param->shape_[i] = shape.at(i);
  }
  Node::CreateConstTensor(C1NUM, {static_cast<int32_t>(shape.size())}, kNumberTypeInt32, KHWC, "shape", shape.data());
}

int ReshapeM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::ReshapeT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

std::vector<EXPR *> ReshapeM::Grad(EXPR *yt) {
  auto shape_of_x = input(0)->dims();
  auto reshape = NN::Reshape(shape_of_x);
  PushOp(reshape);
  return (*reshape)({yt});
}

static ImportReg reg(schema::PrimitiveType_Reshape, ReturnNode<ReshapeM>);

namespace NN {
Node *Reshape(const std::vector<int> &shape) {
  auto node = new (std::nothrow) ReshapeM(shape);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate reshape node";
    return nullptr;
  }
  node->set_name(Node::UniqueName("Reshape"));
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
