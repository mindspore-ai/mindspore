/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dense.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/matmul_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/matmul.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/addmm.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace pyboost_dense {
std::vector<int64_t> GetTransposePerm(const TensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  size_t size = shape.size();
  std::vector<int64_t> perm(size);
  if (size < kDim2) {
    perm[0] = 0;
    return perm;
  }
  perm[size - kDim1] = static_cast<int64_t>(size - kDim2);
  perm[size - kDim2] = static_cast<int64_t>(size - kDim1);
  for (size_t i = 0; i < size - kDim2; ++i) {
    perm[i] = static_cast<int64_t>(i);
  }
  return perm;
}
}  // namespace pyboost_dense

void DenseAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                          const TensorPtr &weight_tensor, const std::optional<TensorPtr> &bias_tensor) {
  MS_LOG(DEBUG) << "Dense Launch start";
  auto x_type = input_tensor->Dtype();
  auto w_type = weight_tensor->Dtype();
  // all dtypes should be the same
  if (x_type->type_id() != w_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For Dense'"
                            << "', the type of 'input' should be same as 'weight', but got 'input' with type Tensor["
                            << x_type->ToString() << "] and 'weight' with type Tensor[" << w_type->ToString() << "].";
  }
  if (bias_tensor.has_value()) {
    auto b_value = bias_tensor.value();
    auto b_type = b_value->Dtype();
    if (x_type->type_id() != b_type->type_id()) {
      MS_EXCEPTION(TypeError) << "For Dense, all dtypes should be the same, but got 'input' with type Tensor["
                              << x_type->ToString() << "] and 'bias' with type Tensor[" << b_type->ToString() << "].";
    }
    // the scenario of bias.rank >= 2D is not supported currently.
    if (b_value->shape().size() >= kDim2) {
      MS_EXCEPTION(ValueError) << "For Dense, the dim of bias should be equal to 0 or 1"
                               << ", but got dim of bias is " << b_value->shape().size() << ".";
    }
  }
  size_t w_rank = weight_tensor->shape().size();
  if (w_rank != kDim1 && w_rank != kDim2) {
    MS_EXCEPTION(ValueError) << "For Dense, the dim of weight should be equal to 1 or 2"
                             << ", but got dim of weight is " << w_rank << ".";
  }
  size_t x_rank = input_tensor->shape().size();
  if (x_rank < kDim1) {
    MS_EXCEPTION(ValueError) << "For Dense, the dim of input should be at least 1"
                             << ", but got dim of input is " << x_rank << ".";
  }

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto perm = pyboost_dense::GetTransposePerm(weight_tensor);
  auto weight_transposed = transpose(weight_tensor, perm);

  auto input_tensor_shape = input_tensor->shape();
  auto input_tensor_rank = input_tensor_shape.size();

  if (input_tensor_rank == kDim2 && bias_tensor.has_value()) {
    auto bias_tensor_ = bias_tensor.value();
    auto addmm_op = CREATE_PYBOOST_OP(Addmm, device::DeviceType::kAscend);
    const auto beta = std::make_shared<Int64Imm>(1);
    const auto alpha = std::make_shared<Int64Imm>(1);
    auto addmm_out = addmm_op->Call(bias_tensor_, input_tensor, weight_transposed, beta, alpha);
    op->set_outputs({addmm_out});
    MS_LOG(DEBUG) << "Dense Launch end";
    return;
  } else if (bias_tensor.has_value() && (bias_tensor.value()->shape().size() == kDim1 || input_tensor_rank == kDim3)) {
    // reshape 2D
    int64_t flattened_dim = 1;
    for (size_t i = 0; i < input_tensor_rank - 1; ++i) {
      flattened_dim = flattened_dim * input_tensor_shape[i];
    }
    int64_t flattened_vector_size = 2;
    std::vector<int64_t> flattened_vector(flattened_vector_size);
    flattened_vector[kIndex0] = static_cast<int64_t>(flattened_dim);
    flattened_vector[kIndex1] = static_cast<int64_t>(input_tensor_shape[input_tensor_rank - 1]);
    auto inp_reshape = reshape(input_tensor, flattened_vector);
    // addmm
    auto bias_tensor_ = bias_tensor.value();
    auto addmm_op = CREATE_PYBOOST_OP(Addmm, device::DeviceType::kAscend);
    const auto beta = std::make_shared<Int64Imm>(1);
    const auto alpha = std::make_shared<Int64Imm>(1);
    auto addmm_out = addmm_op->Call(bias_tensor_, inp_reshape, weight_transposed, beta, alpha);
    // view update shape
    std::vector<int64_t> out_shape;
    std::transform(input_tensor_shape.begin(), input_tensor_shape.end(), std::back_inserter(out_shape),
                   [](int64_t x) { return x; });
    auto addmm_out_shape = addmm_out->shape();
    out_shape[input_tensor_rank - 1] = static_cast<int64_t>(addmm_out_shape[kIndex1]);

    auto output = view(addmm_out, out_shape);
    op->set_outputs({output});
    MS_LOG(DEBUG) << "Dense Launch end";
    return;
  } else {
    auto matmul_op = CREATE_PYBOOST_OP(MatMulExt, device::DeviceType::kAscend);
    auto matmul_out = matmul_op->Call(input_tensor, weight_transposed);
    if (bias_tensor.has_value()) {
      auto bias_tensor_ = bias_tensor.value();
      auto add_op = CREATE_PYBOOST_OP(Add, device::DeviceType::kAscend);
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
