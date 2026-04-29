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
#include "kernel/ascend/aclnn/pyboost_impl/customize/isclose.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IsCloseAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &other_tensor,
                                         const FP32ImmPtr &rtol, const FP32ImmPtr &atol, const BoolImmPtr &equal_nan) {
  auto output = InferOutput<1>(prim::kPrimIsClose, input_tensor, other_tensor, rtol, atol, equal_nan);

  auto rtol_imm = static_cast<double>(GetValue<float>(rtol));
  auto atol_imm = static_cast<double>(GetValue<float>(atol));
  auto equal_nan_imm = GetValue<bool>(equal_nan);

  TensorToDevice(input_tensor, other_tensor);
  TensorAllocate(output);

  // Async
  DISPATCH_ACLNN(aclnnIsClose, input_tensor, other_tensor, rtol_imm, atol_imm, equal_nan_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
