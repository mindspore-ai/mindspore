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

#include "kernel/ascend/aclnn/pyboost_impl/customize/layer_norm_grad_ext.h"
#include <algorithm>
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> LayerNormGradExtAscendCustomize(
  const TensorPtr &dy_tensor, const TensorPtr &x_tensor, const ValueTuplePtr &normalized_shape,
  const TensorPtr &mean_tensor, const TensorPtr &variance_tensor, const TensorPtr &gamma_tensor,
  const TensorPtr &beta_tensor, const ValueTuplePtr &output_mask) {
  MS_LOG(DEBUG) << "Call start";
  // Convert ValuePtr to c++ scalr
  auto outputs = InferOutput<3>(prim::kPrimLayerNormGradExt, dy_tensor, x_tensor, normalized_shape, mean_tensor,
                                variance_tensor, gamma_tensor, beta_tensor, output_mask);

  std::vector<int64_t> normalized_shape_vector = ConvertValueTupleToVector<int64_t>(normalized_shape);
  std::vector<int64_t> output_mask_vector = ConvertValueTupleToVector<int64_t>(output_mask);
  std::vector<uint8_t> output_mask_u8_vec;
  std::transform(output_mask_vector.begin(), output_mask_vector.end(), std::back_inserter(output_mask_u8_vec),
                 [](const int64_t &value) { return static_cast<uint8_t>(value); });

  TensorToDevice(dy_tensor, x_tensor, mean_tensor, variance_tensor, gamma_tensor, beta_tensor);
  TensorAllocate(outputs);

  DISPATCH_ACLNN(aclnnLayerNormBackward, dy_tensor, x_tensor, normalized_shape_vector, mean_tensor, variance_tensor,
                 gamma_tensor, beta_tensor, output_mask_u8_vec, std::get<0>(outputs), std::get<1>(outputs),
                 std::get<2>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
