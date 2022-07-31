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

#include "src/expression/ops/reduce.h"
#include <functional>
#include "src/expression/ops/tile.h"
#include "src/expression/ops/reshape.h"
#include "src/expression/ops/arithmetic.h"
#include "src/expression/ops.h"
#include "src/expression/ops_utils.h"
#include "src/expression/import.h"
#include "nnacl/reduce_parameter.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
ReduceM::ReduceM(schema::ReduceMode mode, bool keep_dims, const std::vector<int> &axis) : Node() {
  expr()->SetSize(C2NUM);
  ReduceParameter *param = reinterpret_cast<ReduceParameter *>(calloc(1, sizeof(ReduceParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  param->mode_ = mode;
  param->keep_dims_ = keep_dims;
  param->reduce_to_end_ = false;
  param->coeff = 1.f;
  SetOpParam(param);
  set_name(UniqueName("Reduce"));
  set_primitive(schema::PrimitiveType_ReduceFusion);
  Node::CreateConstTensor(C1NUM, {static_cast<int32_t>(axis.size())}, kNumberTypeInt32, KHWC, "axis", axis.data());
}

int ReduceM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto reduce_param = reinterpret_cast<const ReduceParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::ReduceFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->keep_dims = reduce_param->keep_dims_;
  prim->mode = static_cast<schema::ReduceMode>(reduce_param->mode_);
  prim->coeff = reduce_param->coeff;
  prim->reduce_to_end = reduce_param->reduce_to_end_;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

std::vector<EXPR *> ReduceM::Grad(EXPR *yt) {
  auto shape_of_x = input(0)->dims();
  std::vector<int> shape_of_axis;

  auto data = input(1)->node()->data()->data().data();
  int size = input(1)->dims().at(0);
  auto int_data = reinterpret_cast<int *>(data);
  for (int i = 0; i < size; i++) {
    shape_of_axis.push_back(int_data[i]);
  }

  // assume no dynamic shape
  ShapeReduce reduce_shape;
  auto output_shape_kept_dims = ShapeReduce()(shape_of_x, shape_of_axis);
  auto tile_scaling = VectorDiv()(shape_of_x, output_shape_kept_dims);
  auto reshape = NN::Reshape(output_shape_kept_dims);
  PushOp(reshape);
  reshape->set_name(name() + "/reshape");
  auto g = (*reshape)({yt}).front();
  auto tile = NN::Tile(tile_scaling);
  PushOp(tile);
  tile->set_name(name() + "/tile");
  auto sum_grad = (*tile)({g}).front();
  auto reduce_param = reinterpret_cast<const ReduceParameter *>(OpParam());
  if (reduce_param->mode_ == schema::ReduceMode_ReduceSum) {
    return {sum_grad};
  } else if (reduce_param->mode_ == schema::ReduceMode_ReduceMean) {
    auto shape_of_y = output(0)->dims();
    auto shape_x_mul = std::accumulate(shape_of_x.begin(), shape_of_x.end(), 1, std::multiplies<int>());
    auto shape_y_mul = std::accumulate(shape_of_y.begin(), shape_of_y.end(), 1, std::multiplies<int>());
    auto div_shape = static_cast<float>(shape_x_mul) / static_cast<float>(shape_y_mul);
    auto div_op = NN::Div();
    PushOp(div_op);
    auto d = div_op->CreateConstTensor(C1NUM, {1}, kNumberTypeFloat32, KHWC, "div_shape", &div_shape);
    auto dx = (*div_op)({sum_grad, d->expr()});
    return dx;
  } else {
    return {};
  }
}

static ImportReg reg(schema::PrimitiveType_ReduceFusion, ReturnNode<ReduceM>);

namespace NN {
Node *ReduceSum(bool keep_dims, const std::vector<int> &axis) {
  auto node = new (std::nothrow) ReduceM(schema::ReduceMode_ReduceSum, keep_dims, axis);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate  reduce sum node";
    return nullptr;
  }
  node->set_name(Node::UniqueName("ReduceSum"));
  return node;
}
Node *ReduceMean(bool keep_dims, const std::vector<int> &axis) {
  auto node = new (std::nothrow) ReduceM(schema::ReduceMode_ReduceMean, keep_dims, axis);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate reduce mean node";
    return nullptr;
  }
  node->set_name(Node::UniqueName("ReduceMean"));
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
