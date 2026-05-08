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

#include "kernel/ascend/aclnn/pyboost_impl/customize/kv_cache_scatter_update.h"
#include <memory>
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr KVCacheScatterUpdateAscendCustomize(const tensor::TensorPtr &var_tensor,
                                                      const tensor::TensorPtr &indices_tensor,
                                                      const tensor::TensorPtr &updates_tensor, const Int64ImmPtr &axis,
                                                      const Int64ImmPtr &reduce) {
  auto axis_imm = GetValue<int64_t>(axis);
  TensorToDevice(var_tensor, indices_tensor, updates_tensor);

  DISPATCH_ACLNN(aclnnInplaceScatterUpdate, var_tensor, indices_tensor, updates_tensor, axis_imm);
  return var_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
