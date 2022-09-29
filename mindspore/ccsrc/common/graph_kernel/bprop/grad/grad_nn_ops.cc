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
#include "common/graph_kernel/bprop/bprop_irbuilder.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER(kTopKOpName).SetBody([](const BpropIRBuilder *builder) -> NodePtrList {
  auto input_x = builder->GetInput(kIndex0);
  auto out = builder->GetInput(kIndex2);
  auto dout = builder->GetInput(kIndex3);

  auto indices = builder->TupleGetItem(out, kIndex1);
  auto dout0 = builder->TupleGetItem(dout, kIndex0);

  auto in_shape = builder->GetShape(input_x);
  auto in_lastdim = in_shape.back();

  auto ind_shape = builder->GetShape(indices);
  auto ind_lastdim = ind_shape.back();

  auto ind_2d = builder->Reshape(indices, {-1, ind_lastdim});
  auto outerdim = builder->GetShape(ind_2d)[0];  // k

  // [0, outerdim, 2*outerdim, ..., (k-1)*outerdim]
  auto indices_dtype = builder->GetDtype(indices);
  std::vector<int64_t> range_flatten_index_vec(outerdim);
  for (int64_t i = 0; i < outerdim; i++) {
    range_flatten_index_vec[i] = i * in_lastdim;
  }
  auto range_flatten_index = builder->Tensor(range_flatten_index_vec);
  if (indices_dtype->type_id() != kNumberTypeInt64) {
    range_flatten_index = builder->Cast(range_flatten_index, indices_dtype);
  }

  auto ind = builder->Reshape(ind_2d + range_flatten_index, {-1, 1});
  auto in_shape_1d = ShapeVector(1, std::accumulate(in_shape.begin(), in_shape.end(), 1, std::multiplies<int64_t>()));
  auto out_grad = builder->Emit("ScatterNd", {ind, builder->Reshape(dout0, {-1}), builder->Tensor(in_shape_1d)});
  out_grad = builder->Reshape(out_grad, in_shape);

  auto grad_k = builder->ZerosLike(builder->GetInput(kIndex1));
  return {out_grad, grad_k};
});
}  // namespace mindspore::expander::bprop
