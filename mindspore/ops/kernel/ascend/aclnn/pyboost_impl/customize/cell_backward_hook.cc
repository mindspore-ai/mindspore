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

#include "kernel/ascend/aclnn/pyboost_impl/customize/cell_backward_hook.h"
#include <memory>
#include "include/pynative/utils/pyboost/customize/cell_backward_hook.h"

namespace mindspore::kernel::pyboost {
void CellBackwardHookAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &tensors_list) {
  MS_LOG(DEBUG) << "Cell BackwardHook Ascend start";
  CellBackwardHookCustomize(op, tensors_list);
  MS_LOG(DEBUG) << "Cell BackwardHook Ascend end";
}
}  // namespace mindspore::kernel::pyboost
