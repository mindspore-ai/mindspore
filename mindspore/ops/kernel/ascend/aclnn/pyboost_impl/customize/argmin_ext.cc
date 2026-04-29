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

#include "kernel/ascend/aclnn/pyboost_impl/customize/argmin_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ArgMinAscendCustomize(const TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &dim,
                                        const BoolImmPtr &keepdim) {
  MS_LOG(DEBUG) << "aclnnArgmin call start";
  auto output = InferOutput<1>(prim::kPrimArgMinExt, input_tensor, dim, keepdim);

  int64_t real_dim = 0;
  auto real_keepdim = false;
  TensorPtr real_input;
  if (dim.has_value()) {
    real_dim = GetValue<int64_t>(dim.value());
    real_keepdim = GetValue<bool>(keepdim);
    real_input = input_tensor;
  } else {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    real_input = reshape(input_tensor, {-1});
  }

  TensorToDevice(real_input);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnArgMin, real_input, real_dim, real_keepdim, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
