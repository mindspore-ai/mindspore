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

#include "plugin/device/ascend/kernel/pyboost/customize/transpose.h"

#include <memory>
#include <vector>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/pynative/op_executor.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel::pyboost {
void TransposeAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_x_tensor,
                              const ValueTuplePtr &dims) {
  MS_LOG(EXCEPTION) << "Not implement transpose ascend customize function.";
}
}  // namespace mindspore::kernel::pyboost
