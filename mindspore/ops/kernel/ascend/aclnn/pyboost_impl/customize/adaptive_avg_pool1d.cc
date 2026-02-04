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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_avg_pool1d.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/auto_generate/adaptive_avg_pool2d_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int kShape2dDims = 2;
}
tensor::TensorPtr AdaptiveAvgPool1DAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_x_tensor,
                                                   const ValueTuplePtr &output_size) {
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  OpRunner::InferOpOutput(op, input_x_tensor, output_size);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto input_shape = input_x_tensor->shape();
  auto origin_shape_dim = SizeToLong(input_shape.size());

  // unsqueeze shape
  std::vector<int64_t> expand_input_shape;
  for (auto i = 0; i < origin_shape_dim - 1; i++) {
    expand_input_shape.emplace_back(input_shape[i]);
  }
  expand_input_shape.emplace_back(1);
  expand_input_shape.emplace_back(input_shape[origin_shape_dim - 1]);
  auto input_x_imm = reshape(input_x_tensor, expand_input_shape);

  auto output_size_val = ConvertValueTupleToVector<int64_t>(output_size);
  auto output_size_2d = std::make_shared<ValueTuple>(
    std::vector<ValuePtr>{std::make_shared<Int64Imm>(1), std::make_shared<Int64Imm>(output_size_val[0])});
  // call AdaptiveAvgPool2dExt
  auto adaptive_avg_pool2d_op = CREATE_PYBOOST_OP(AdaptiveAvgPool2DExt, device::DeviceType::kAscend);
  auto output_adaptive_avg_pool2d_tensor = adaptive_avg_pool2d_op->Call(input_x_imm, output_size_2d);

  // squeeze shape
  auto shape_pool2d = output_adaptive_avg_pool2d_tensor->shape();
  auto shape_pool2d_dim = SizeToLong(shape_pool2d.size());
  std::vector<int64_t> squeeze_input_shape;
  if (shape_pool2d_dim <= kShape2dDims) {
    MS_LOG(EXCEPTION) << "For AdaptiveAvgPool1DAscendCustomize, the value of shape_pool2d.size is invalid.";
  }
  constexpr int offset = 2;
  for (auto i = 0; i < shape_pool2d_dim - offset; i++) {
    squeeze_input_shape.emplace_back(shape_pool2d[i]);
  }
  squeeze_input_shape.emplace_back(shape_pool2d[shape_pool2d_dim - 1]);
  auto output_tensor = reshape(output_adaptive_avg_pool2d_tensor, squeeze_input_shape);
  op->set_outputs({output_tensor});
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
