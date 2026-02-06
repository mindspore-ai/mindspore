/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/masked_scatter.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/contiguous.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

namespace {

tensor::TensorPtr MaskedScatterAscendCall(const TensorPtr &x_tensor_bd, const TensorPtr &mask_tensor_bd,
                                          const TensorPtr &updates_tensor, const TensorPtr &output_tensor) {
  DISPATCH_ACLNN(aclnnInplaceCopy, output_tensor, x_tensor_bd);
  DISPATCH_ACLNN(aclnnInplaceMaskedScatter, output_tensor, mask_tensor_bd, updates_tensor);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedScatterAscendCustomize(const TensorPtr &x_tensor, const TensorPtr &mask_tensor,
                                               const TensorPtr &updates_tensor) {
  const std::vector<int64_t> &input_shape = x_tensor->shape();
  const std::vector<int64_t> &target_shape = mask_tensor->shape();
  std::vector<int64_t> expand_shape = ops::CalBroadCastShapeV3(input_shape, target_shape);

  const ValueTuplePtr &expand_shape_ptr = ops::ConvertShapeVectorToValueTuple(expand_shape);
  MS_EXCEPTION_IF_NULL(expand_shape_ptr);

  TensorPtr x_tensor_bd = x_tensor;
  TensorPtr mask_tensor_bd = mask_tensor;

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  if (input_shape != expand_shape) {
    x_tensor_bd = broadcast_to_view(x_tensor, expand_shape);
  }
  if (target_shape != expand_shape) {
    mask_tensor_bd = broadcast_to_view(mask_tensor, expand_shape);
  }

  auto output = InferOutput<1>(prim::kPrimMaskedScatter, x_tensor_bd, mask_tensor_bd, updates_tensor);
  TensorToDevice(x_tensor_bd, mask_tensor_bd, updates_tensor);
  TensorAllocate(output);
  MaskedScatterAscendCall(x_tensor_bd, mask_tensor_bd, updates_tensor, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
