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

#include "plugin/device/ascend/kernel/pyboost/customize/dense.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/auto_generate/transpose.h"
#include "plugin/device/ascend/kernel/pyboost/auto_generate/matmul_ext.h"
#include "plugin/device/ascend/kernel/pyboost/auto_generate/addmm.h"
#include "plugin/device/ascend/kernel/pyboost/auto_generate/add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
ValueTuplePtr GetTransposePerm(const BaseTensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  size_t size = shape.size();
  std::vector<ValuePtr> perm(size);
  if (size < kDim2) {
    auto zero = std::make_shared<Int64Imm>(0);
    perm[0] = MakeValue(zero);
    return std::make_shared<ValueTuple>(perm);
  }
  perm[size - kDim1] = MakeValue(static_cast<int64_t>(size - kDim2));
  perm[size - kDim2] = MakeValue(static_cast<int64_t>(size - kDim1));
  for (size_t i = 0; i < size - kDim2; ++i) {
    perm[i] = MakeValue(static_cast<int64_t>(i));
  }
  return std::make_shared<ValueTuple>(perm);
}
}  // namespace

void DenseAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                          const BaseTensorPtr &weight_tensor, const std::optional<BaseTensorPtr> &bias_tensor) {
  MS_LOG(DEBUG) << "Dense Launch start";
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, bias_tensor);
  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;

  auto perm = GetTransposePerm(weight_tensor);
  auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
  auto weight_transposed = transpose_op->Call(weight_tensor, perm);

  auto input_tensor_rank = input_tensor->shape().size();

  if (input_tensor_rank == kDim2 && bias_tensor.has_value()) {
    auto bias_tensor_ = bias_tensor.value();
    auto addmm_op = CREATE_PYBOOST_OP(Addmm, device_name);
    const auto beta = std::make_shared<Int64Imm>(1);
    const auto alpha = std::make_shared<Int64Imm>(1);
    auto addmm_out = addmm_op->Call(bias_tensor_, input_tensor, weight_transposed, beta, alpha);
    op->set_outputs({addmm_out});
    MS_LOG(DEBUG) << "Dense Launch end";
    return;
  } else {
    auto matmul_op = CREATE_PYBOOST_OP(MatMulExt, device_name);
    auto matmul_out = matmul_op->Call(input_tensor, weight_transposed);
    if (bias_tensor.has_value()) {
      auto bias_tensor_ = bias_tensor.value();
      auto add_op = CREATE_PYBOOST_OP(Add, device_name);
      auto add_out = add_op->Call(matmul_out, bias_tensor_);
      op->set_outputs({add_out});
      MS_LOG(DEBUG) << "Dense Launch end";
      return;
    }
    op->set_outputs({matmul_out});
    MS_LOG(DEBUG) << "Dense Launch end";
    return;
  }
}  // namespace pyboost
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
