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
#include "mindspore/ops/kernel/cpu/pyboost/customize/divmods.h"
#include <memory>
#include <utility>
#include "include/pynative/utils/pyboost/customize/divmod.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DivModsCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, const ScalarPtr &y_scalar,
                         const std::optional<Int64ImmPtr> &rounding_mode) {
  MS_LOG(DEBUG) << "DivMods Call start";
  OpRunner::InferOpOutput(op, x_tensor, y_scalar, rounding_mode);

  const auto &device_name = device::DeviceType::kCPU;

  // handle type promotion manually since the CPU kernelmod DivMods does not support it
  const auto out_dtype = op->output(0)->Dtype();
  auto x_tensor_cast = x_tensor;
  if (x_tensor->Dtype()->type_id() != out_dtype->type_id()) {
    x_tensor_cast = PyBoostUtils::CastTensor(x_tensor, out_dtype->type_id(), device_name);
  }
  const auto y_tensor = PyBoostUtils::ScalarToTensor(y_scalar, out_dtype);
  const auto out = DivModCustomize(op, x_tensor_cast, y_tensor, rounding_mode);
  MS_LOG(DEBUG) << "DivMods Call end";

  op->set_outputs({out});
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
