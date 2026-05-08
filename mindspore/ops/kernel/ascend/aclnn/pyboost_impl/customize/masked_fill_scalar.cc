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

#include "kernel/ascend/aclnn/pyboost_impl/customize/masked_fill_scalar.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
tensor::TensorPtr MaskedFillScalarAscendCall(const TensorPtr &input_tensor, const TensorPtr &mask_tensor,
                                             const ScalarPtr &value, const TensorPtr &output_tensor) {
  DISPATCH_ACLNN(aclnnInplaceCopy, output_tensor, input_tensor);
  DISPATCH_ACLNN(aclnnInplaceMaskedFillScalar, output_tensor, mask_tensor, value);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedFillScalarAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &mask_tensor,
                                                  const ScalarPtr &value) {
  auto output = InferOutput<1>(prim::kPrimMaskedFillScalar, input_tensor, mask_tensor, value);
  TensorToDevice(input_tensor, mask_tensor);
  TensorMalloc(output);
  MaskedFillScalarAscendCall(input_tensor, mask_tensor, value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
