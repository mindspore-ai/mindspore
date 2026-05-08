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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_clamp_tensor.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceClampTensorAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                    const std::optional<tensor::TensorPtr> &min_tensor,
                                                    const std::optional<tensor::TensorPtr> &max_tensor) {
  MS_LOG(DEBUG) << "Call InplaceClampTensor start";
  InferOutput<1>(prim::kPrimInplaceClampTensor, input_tensor, min_tensor, max_tensor);
  TensorToDevice(input_tensor, min_tensor, max_tensor);

  DISPATCH_ACLNN(aclnnClampTensor, input_tensor, min_tensor, max_tensor, input_tensor);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
