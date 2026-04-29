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

#include "kernel/ascend/aclnn/pyboost_impl/customize/muls.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MulsAscendCustomize(const TensorPtr &input_tensor, const ScalarPtr &other_scalar) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(other_scalar);
  auto output = InferOutput<1>(prim::kPrimMuls, input_tensor, other_scalar);
  // No need to convert input
  TensorToDevice(input_tensor);
  TensorAllocate(output);

  ScalarPtr other_scalar_real = ops::FetchRealScalar(other_scalar);

  // Async
  DISPATCH_ACLNN(aclnnMuls, input_tensor, other_scalar_real, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
