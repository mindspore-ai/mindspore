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

#include "kernel/ascend/aclnn/pyboost_impl/customize/pow_scalar_tensor.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr PowScalarTensorAscendCustomize(const ScalarPtr &input_scalar, const TensorPtr &exponent_tensor) {
  MS_EXCEPTION_IF_NULL(input_scalar);
  MS_EXCEPTION_IF_NULL(exponent_tensor);
  auto output = InferOutput<1>(prim::kPrimPowScalarTensor, input_scalar, exponent_tensor);

  ScalarPtr input_scalar_real = ops::FetchRealScalar(input_scalar);
  TensorToDevice(exponent_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnPowScalarTensor, input_scalar_real, exponent_tensor, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
