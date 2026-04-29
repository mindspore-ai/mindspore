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

#include "kernel/ascend/aclnn/pyboost_impl/customize/layer_norm_ext.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> LayerNormExtAscendCustomize(
  const TensorPtr &input_tensor, const ValueTuplePtr &normalized_shape,
  const std::optional<TensorPtr> &weight_opt_tensor, const std::optional<TensorPtr> &bias_opt_tensor,
  const FP32ImmPtr &eps) {
  MS_LOG(DEBUG) << "Call start";

  // Convert ValuePtr to c++ scalr
  auto outputs =
    InferOutput<3>(prim::kPrimLayerNormExt, input_tensor, normalized_shape, weight_opt_tensor, bias_opt_tensor, eps);

  std::vector<int64_t> normalized_shape_vector = ConvertValueTupleToVector<int64_t>(normalized_shape);
  auto eps_imm = static_cast<double>(GetValue<float>(eps));

  TensorToDevice(input_tensor, weight_opt_tensor, bias_opt_tensor);
  TensorAllocate(outputs);

  DISPATCH_ACLNN(aclnnLayerNorm, input_tensor, normalized_shape_vector, weight_opt_tensor, bias_opt_tensor, eps_imm,
                 std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
