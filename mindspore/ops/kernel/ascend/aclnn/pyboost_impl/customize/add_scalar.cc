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

#include "kernel/ascend/aclnn/pyboost_impl/customize/add_scalar.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AddScalarAscendCustomize(const TensorPtr &input_tensor, const ScalarPtr &other_scalar,
                                           const ScalarPtr &alpha_scalar) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(other_scalar);
  MS_EXCEPTION_IF_NULL(alpha_scalar);
  auto output = InferOutput<1>(prim::kPrimAddScalar, input_tensor, other_scalar, alpha_scalar);
  // No need to convert input
  TensorToDevice(input_tensor);
  TensorAllocate(output);

  auto other_scalar_real = ops::FetchRealScalar(other_scalar);
  auto alpha_scalar_real = ops::FetchRealScalar(alpha_scalar);
  // Async
  DISPATCH_ACLNN(aclnnAdds, input_tensor, other_scalar_real, alpha_scalar_real, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
