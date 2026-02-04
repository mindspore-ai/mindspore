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
#include "kernel/gpu/pyboost/customize/divmod.h"
#include <memory>
#include <utility>
#include "include/pynative/utils/pyboost/customize/divmod.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DivModGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                                     const TensorPtr &y_tensor, const std::optional<Int64ImmPtr> &rounding_mode) {
  DivModCustomize(op, x_tensor, y_tensor, rounding_mode);
  auto sync = runtime::RuntimeConf::GetInstance()->launch_blocking();
  if (sync && !op->device_context()->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for op DivMod.";
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
