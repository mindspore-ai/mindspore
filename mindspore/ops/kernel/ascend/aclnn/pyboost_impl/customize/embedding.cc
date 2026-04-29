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

#include "kernel/ascend/aclnn/pyboost_impl/customize/embedding.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void InplaceEmbeddingRenormCall(const TensorPtr &input, const TensorPtr &weight,
                                const std::optional<FP32ImmPtr> &max_norm, const FP32ImmPtr &norm_type) {
  if (!max_norm.has_value()) {
    return;
  }

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  TensorPtr copy_input = nullptr;
  auto copy_op = CREATE_PYBOOST_OP(Copy, device::DeviceType::kAscend);
  copy_input = copy_op->Call(input);
  copy_input = view(copy_input, {-1});

  TensorToDevice(copy_input, weight);
  MS_EXCEPTION_IF_NULL(max_norm.value());
  auto max_norm_value = static_cast<double>(max_norm.value()->value());
  auto norm_type_value = static_cast<double>(norm_type->value());
  DISPATCH_ACLNN(aclnnEmbeddingRenorm, weight, copy_input, max_norm_value, norm_type_value);
}
}  // namespace

tensor::TensorPtr EmbeddingAscendCustomize(const TensorPtr &input, const TensorPtr &weight,
                                           const std::optional<Int64ImmPtr> &padding_idx,
                                           const std::optional<FP32ImmPtr> &max_norm, const FP32ImmPtr &norm_type,
                                           const BoolImmPtr &scale_grad_by_freq) {
  auto output =
    InferOutput<1>(prim::kPrimEmbedding, input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq);

  InplaceEmbeddingRenormCall(input, weight, max_norm, norm_type);

  TensorToDevice(input, weight);
  TensorAllocate(output);

  // Async
  DISPATCH_ACLNN(aclnnEmbedding, weight, input, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
