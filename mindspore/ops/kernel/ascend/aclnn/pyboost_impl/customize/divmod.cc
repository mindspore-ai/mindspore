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

#include "kernel/ascend/aclnn/pyboost_impl/customize/divmod.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DivModAscendCustomize(const TensorPtr &x_tensor, const TensorPtr &y_tensor,
                                        const std::optional<Int64ImmPtr> &rounding_mode) {
  auto output = InferOutput<1>(prim::kPrimDivMod, x_tensor, y_tensor, rounding_mode);

  auto mode = 0;
  if (rounding_mode.has_value()) {
    mode = GetValue<int64_t>(rounding_mode.value());
  }

  TensorToDevice(x_tensor, y_tensor);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnDivMod, x_tensor, y_tensor, mode, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
