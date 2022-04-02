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
#include <memory>
#include "common/graph_kernel/model/graph_builder.h"
#include "common/graph_kernel/expanders/expander_factory.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::inner {
NodePtr GraphBuilder::Reshape(const NodePtr &input, const ShapeVector &shape) const {
  auto shape_value = MakeValue(shape);
  return Emit("Reshape", {input}, {{"shape", shape_value}});
}
NodePtr GraphBuilder::BroadcastTo(const NodePtr &input, const ShapeVector &shape) const {
  auto shape_value = MakeValue(shape);
  return Emit("BroadcastTo", {input}, {{"shape", shape_value}});
}

NodePtr GraphBuilder::ReduceSum(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims) const {
  auto reduce_axis = MakeValue(axis);
  auto keep_dims_value = MakeValue(keep_dims);
  return Emit("ReduceSum", {input}, {{"axis", reduce_axis}, {"keep_dims", keep_dims_value}});
}
NodePtr GraphBuilder::ReduceMax(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims) const {
  auto reduce_axis = MakeValue(axis);
  auto keep_dims_value = MakeValue(keep_dims);
  return Emit("ReduceMax", {input}, {{"axis", reduce_axis}, {"keep_dims", keep_dims_value}});
}
}  // namespace mindspore::graphkernel::inner
