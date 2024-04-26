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
#include "plugin/device/cpu/kernel/pyboost/customize/divmod.h"
#include <memory>
#include <utility>
#include "mindspore/ccsrc/kernel/pyboost/customize/divmod.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr DivModCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                         const BaseTensorPtr &y_tensor,
                                         const std::optional<Int64ImmPtr> &rounding_mode) {
  DivModCustomize(op, x_tensor, y_tensor, rounding_mode);
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
