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
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"

#include <vector>

#include "utils/ms_context.h"
#include "mindapi/base/type_id.h"
#include "mindapi/base/shape_vector.h"
#include "backend/ms_backend/graph_fusion/model/lite_graph.h"

namespace mindspore::graphkernel::inner {
NodePtr GraphBuilder::Reshape(const NodePtr &input, const ShapeVector &shape) const {
  auto shape_tensor = Tensor(shape);
  return Emit("Reshape", {input, shape_tensor});
}

NodePtr GraphBuilder::BroadcastTo(const NodePtr &input, const ShapeVector &shape) const {
  auto shape_value = Tensor(shape);
  return Emit("BroadcastTo", {input, shape_value});
}

NodePtr GraphBuilder::Gather(const NodePtr &param, const NodePtr &indice, int64_t axis, int64_t batch_dims) const {
  auto axis_tensor = Tensor(axis, kNumberTypeInt64);
  return Emit("Gather", {param, indice, axis_tensor}, {{"batch_dims", MakeValue(batch_dims)}});
}

NodePtr GraphBuilder::Concat(const NodePtrList &inputs, const int64_t &axis) const {
  auto axis_value = MakeValue(axis);
  return Emit("Concat", inputs, {{"axis", axis_value}});
}

NodePtr GraphBuilder::Transpose(const NodePtr &input, const ShapeVector &perm) const {
  auto perm_tensor = Tensor(perm);
  return Emit("Transpose", {input, perm_tensor});
}

NodePtr GraphBuilder::ReduceSum(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims) const {
  auto reduce_axis = Tensor(axis);
  auto keep_dims_value = MakeValue(keep_dims);
  return Emit("ReduceSum", {input, reduce_axis}, {{"keep_dims", keep_dims_value}, {"skip_mode", MakeValue(false)}});
}
NodePtr GraphBuilder::ReduceMax(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims) const {
  auto reduce_axis = Tensor(axis);
  auto keep_dims_value = MakeValue(keep_dims);
  return Emit("ReduceMax", {input, reduce_axis}, {{"keep_dims", keep_dims_value}});
}
NodePtr GraphBuilder::ReduceMin(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims) const {
  auto reduce_axis = Tensor(axis);
  auto keep_dims_value = MakeValue(keep_dims);
  return Emit("ReduceMin", {input, reduce_axis}, {{"keep_dims", keep_dims_value}});
}

NodePtr GraphBuilder::TupleGetItem(const NodePtr &input, int64_t index) const {
  auto index_node = Scalar(index);
  return Emit("TupleGetItem", {input, index_node});
}

NodePtr GraphBuilder::StridedSlice(const NodePtr &input, const std::vector<int64_t> &begin,
                                   const std::vector<int64_t> &end, const std::vector<int64_t> &strides) const {
  auto begin_node = Tensor(begin);
  auto end_node = Tensor(end);
  auto strides_node = Tensor(strides);
  return Emit("StridedSlice", {input, begin_node, end_node, strides_node},
              {{"shrink_axis_mask", MakeValue(static_cast<int64_t>(0))},
               {"begin_mask", MakeValue(static_cast<int64_t>(0))},
               {"ellipsis_mask", MakeValue(static_cast<int64_t>(0))},
               {"new_axis_mask", MakeValue(static_cast<int64_t>(0))},
               {"end_mask", MakeValue(static_cast<int64_t>(0))}});
}
NodePtr GraphBuilder::Tanh(const NodePtr &input) const {
  auto device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kAscendDevice) {
    // Tanh(x) = 1 - 2/(e^{2x}+1)
    auto tanh_exp = Exp(Mul(input, Tensor(2, input->type)));
    auto tanh_add_0 = Add(tanh_exp, Tensor(1, input->type));
    auto tanh_rec = Reciprocal(tanh_add_0);
    auto tanh_neg = Mul(tanh_rec, Tensor(-2, input->type));
    auto tanh_add_1 = Add(tanh_neg, Tensor(1, input->type));
    return tanh_add_1;
  } else {
    return Emit("Tanh", {input});
  }
}
}  // namespace mindspore::graphkernel::inner
