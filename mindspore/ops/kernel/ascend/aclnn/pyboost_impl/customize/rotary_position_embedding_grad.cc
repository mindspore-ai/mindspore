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

#include "kernel/ascend/aclnn/pyboost_impl/customize/rotary_position_embedding_grad.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> RotaryPositionEmbeddingGradAscendCustomize(
  const TensorPtr &dy_tensor, const TensorPtr &cos_tensor, const TensorPtr &sin_tensor,
  const std::optional<TensorPtr> &dx_tensor, const Int64ImmPtr &mode) {
  auto outputs =
    InferOutput<3>(prim::kPrimRotaryPositionEmbeddingGrad, dy_tensor, cos_tensor, sin_tensor, dx_tensor, mode);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto mode_imm = GetValue<int64_t>(mode);
  TensorToDevice(dy_tensor, cos_tensor, sin_tensor, dx_tensor);
  TensorAllocate(outputs);
  auto [o0, o1, o2] = outputs;

  if (dx_tensor.has_value()) {
    DISPATCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, dy_tensor, cos_tensor, sin_tensor, dx_tensor.value(), mode_imm, o0,
                   o1, o2);
  } else {
    DISPATCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, dy_tensor, cos_tensor, sin_tensor, nullptr, mode_imm, o0, o1, o2);
  }
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
