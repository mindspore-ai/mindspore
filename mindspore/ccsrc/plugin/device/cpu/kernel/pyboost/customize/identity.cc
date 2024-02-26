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

#include "plugin/device/cpu/kernel/pyboost/customize/identity.h"
#include <memory>
#include <utility>
#include "mindspore/ccsrc/kernel/pyboost/customize/identity.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IdentityCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  MS_LOG(DEBUG) << "Identity call start";
  IdentityCustomize(op, x_tensor);
  MS_LOG(DEBUG) << "Identity call end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
