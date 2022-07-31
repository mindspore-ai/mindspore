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

#include "src/expression/ops/softmax.h"
#include "nnacl/softmax_parameter.h"
#include "inner/model_generated.h"
#include "src/expression/import.h"
#include "src/expression/ops/reshape.h"
#include "src/expression/ops/reduce.h"
#include "src/expression/ops/arithmetic.h"
#include "src/expression/ops/transpose.h"
#include "src/litert/cxx_api/expression/node_impl.h"
#include "src/expression/ops.h"

namespace mindspore {
namespace lite {
SoftmaxM::SoftmaxM(int axis) {
  auto param = reinterpret_cast<SoftmaxParameter *>(calloc(1, sizeof(SoftmaxParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  param->axis_ = axis;
  SetOpParam(param);
  set_primitive(schema::PrimitiveType_Softmax);
  set_name(UniqueName("Softmax"));
}

std::vector<int> SoftmaxM::getTransposeAxis(const std::vector<int> &shape, int axis) {
  int rank = shape.size();
  if (axis < 0) {
    axis += rank;
  }
  std::vector<int> reverse_axis(rank);
  std::iota(reverse_axis.begin(), reverse_axis.end(), 0);
  reverse_axis.at(axis) = rank - 1;
  reverse_axis.at(rank - 1) = axis;
  return reverse_axis;
}

std::vector<EXPR *> SoftmaxM::Grad(EXPR *yt) {
  auto x = input(0);
  auto out = output(0);
  auto shape_of_x = x->dims();
  auto param = reinterpret_cast<const SoftmaxParameter *>(OpParam());
  auto reverse_axis = getTransposeAxis(shape_of_x, param->axis_);

  auto transpose_out = NN::Transpose(reverse_axis);
  transpose_out->set_name(kGradName + "/" + name() + "/" + transpose_out->name() + "/out/");
  PushOp(transpose_out);
  auto y_trn = (*transpose_out)({out}).front();

  auto transpose_dout = NN::Transpose(reverse_axis);
  transpose_dout->set_name(kGradName + "/" + name() + "/" + transpose_dout->name() + "/dout/");
  PushOp(transpose_dout);
  auto yt_trn = (*transpose_dout)({yt}).front();

  auto mul0 = NN::Mul();
  mul0->set_name(kGradName + "/" + name() + "/" + mul0->name() + "0");
  PushOp(mul0);
  auto tmp0 = (*mul0)({y_trn, yt_trn}).front();

  auto sum_func = NN::ReduceSum(true, {-1});
  sum_func->set_name(kGradName + "/" + name() + "/" + sum_func->name());
  PushOp(sum_func);
  auto tmp1 = (*sum_func)({tmp0}).front();

  auto sub = NN::Sub();
  sub->set_name(kGradName + "/" + name() + "/" + sub->name());
  PushOp(sub);
  auto tmp2 = (*sub)({yt_trn, tmp1}).front();

  auto mul1 = NN::Mul();
  mul1->set_name(kGradName + "/" + name() + "/" + mul1->name() + "1");
  PushOp(mul1);
  auto tmp3 = (*mul1)({y_trn, tmp2});

  auto transpose_dx = NN::Transpose(reverse_axis);
  transpose_dx->set_name(kGradName + "/" + name() + "/" + transpose_dx->name() + "/dx");
  PushOp(transpose_dx);
  auto dx = (*transpose_dx)({tmp3});
  return dx;
}

int SoftmaxM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const SoftmaxParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::SoftmaxT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->axis.push_back(param->axis_);
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Softmax, ReturnNode<SoftmaxM>);

namespace NN {
Node *Softmax(int axis) {
  auto node = new (std::nothrow) SoftmaxM(axis);
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
