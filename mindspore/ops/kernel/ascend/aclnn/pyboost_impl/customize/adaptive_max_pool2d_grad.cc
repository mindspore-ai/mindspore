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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_max_pool2d_grad.h"
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AdaptiveMaxPool2DGradAscendCustomize(const TensorPtr &y_grad, const TensorPtr &x,
                                                       const TensorPtr &argmax) {
  auto output = InferOutput<1>(prim::kPrimAdaptiveMaxPool2DGrad, y_grad, x, argmax);
  TensorToDevice(y_grad, x, argmax);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnAdaptiveMaxPool2dBackward, y_grad, x, argmax, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
