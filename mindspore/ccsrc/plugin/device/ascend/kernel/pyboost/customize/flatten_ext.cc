/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/customize/flatten_ext.h"
#include <memory>
#include <algorithm>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
ValueTuplePtr ShapeCalc(const BaseTensorPtr &input_x, const int64_t &start_dim, const int64_t &end_dim) {
  const auto &input_shape = input_x->shape();
  int64_t dim_size = SizeToLong(input_shape.size());
  if (dim_size == 0) {
    return std::make_shared<ValueTuple>(std::vector<ValuePtr>({std::make_shared<Int64Imm>(1)}));
  }
  CheckAndConvertUtils::CheckInRange<int64_t>("start_dim", start_dim, kIncludeBoth, {-dim_size, dim_size - 1},
                                              "flatten");
  CheckAndConvertUtils::CheckInRange<int64_t>("end_dim", end_dim, kIncludeBoth, {-dim_size, dim_size - 1}, "flatten");
  auto start_dim_fix = start_dim < 0 ? start_dim + dim_size : start_dim;
  auto end_dim_fix = end_dim < 0 ? end_dim + dim_size : end_dim;
  if (start_dim_fix > end_dim_fix) {
    MS_EXCEPTION(ValueError) << "For 'flatten', 'start_dim' cannot come after 'end_dim'.";
  }
  if (start_dim_fix == end_dim_fix) {
    std::vector<ValuePtr> out_shape;
    std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(out_shape),
                   [](int64_t x) { return MakeValue(x); });
    return std::make_shared<ValueTuple>(out_shape);
  }

  auto begin = input_shape.begin() + start_dim_fix;
  auto end = input_shape.begin() + end_dim_fix + 1;
  auto slice_numel = std::accumulate(begin, end, static_cast<int64_t>(1), std::multiplies<int64_t>());
  std::vector<ValuePtr> shape;
  shape.reserve(dim_size - end_dim_fix + start_dim_fix);
  for (int64_t i = 0; i < start_dim_fix; i++) {
    auto axis = MakeValue(std::make_shared<Int64Imm>(input_shape[i]));
    shape.push_back(axis);
  }
  shape.push_back(MakeValue(std::make_shared<Int64Imm>(slice_numel)));
  for (int64_t i = end_dim_fix + 1; i < dim_size; i++) {
    auto axis = MakeValue(std::make_shared<Int64Imm>(input_shape[i]));
    shape.push_back(axis);
  }
  return std::make_shared<ValueTuple>(shape);
}
}  // namespace

void FlattenExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_x_tensor,
                               const Int64ImmPtr &start_dim, const Int64ImmPtr &end_dim) {
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
  auto start_dim_value = GetValue<int64_t>(start_dim);
  auto end_dim_value = GetValue<int64_t>(end_dim);
  auto new_shape = ShapeCalc(input_x_tensor, start_dim_value, end_dim_value);
  auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
  reshape_op->Call(input_x_tensor, new_shape);
  op->set_input_abs({input_x_tensor->ToAbstract(), start_dim->ToAbstract(), end_dim->ToAbstract()});
  op->set_output_abs(reshape_op->output_abs());
  op->set_outputs(reshape_op->outputs());
  MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
