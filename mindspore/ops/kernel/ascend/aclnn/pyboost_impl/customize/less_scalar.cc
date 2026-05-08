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

#include "kernel/ascend/aclnn/pyboost_impl/customize/less_scalar.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr LessScalarAscendCustomize(const TensorPtr &input_tensor, const ScalarPtr &other_scalar) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(other_scalar);
  auto output = InferOutput<1>(prim::kPrimLessScalar, input_tensor, other_scalar);

  auto other_scalar_real = ops::FetchRealScalar(other_scalar);
  TensorToDevice(input_tensor);
  TensorMalloc(output);

  MS_LOG(DEBUG) << "Run device task LessScalar start";
  DISPATCH_ACLNN(aclnnLtScalar, input_tensor, other_scalar_real, output);
  MS_LOG(DEBUG) << "Run device task LessScalar end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
