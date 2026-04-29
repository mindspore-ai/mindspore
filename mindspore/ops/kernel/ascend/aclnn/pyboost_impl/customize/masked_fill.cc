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

#include "kernel/ascend/aclnn/pyboost_impl/customize/masked_fill.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {

tensor::TensorPtr MaskedFillAscendCall(const TensorPtr &input_tensor, const TensorPtr &mask_tensor,
                                       const TensorPtr &value_tensor, const TensorPtr &output_tensor) {
  DISPATCH_ACLNN(aclnnInplaceCopy, output_tensor, input_tensor);
  DISPATCH_ACLNN(aclnnInplaceMaskedFillTensor, output_tensor, mask_tensor, value_tensor);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedFillAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &mask_tensor,
                                            const TensorPtr &value_tensor) {
  auto output = InferOutput<1>(prim::kPrimMaskedFill, input_tensor, mask_tensor, value_tensor);
  TensorToDevice(input_tensor, mask_tensor, value_tensor);
  TensorAllocate(output);
  MaskedFillAscendCall(input_tensor, mask_tensor, value_tensor, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
