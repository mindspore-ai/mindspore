/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/threshold_grad.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ThresholdGradAscendCustomize(const TensorPtr &dy_tensor, const TensorPtr &y_tensor,
                                               const ScalarPtr &threshold) {
  auto output = InferOutput<1>(prim::kPrimThresholdGrad, dy_tensor, y_tensor, threshold);
  // No need to convert input
  TensorToDevice(dy_tensor, y_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnThresholdBackward, dy_tensor, y_tensor, threshold, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
