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

#include "mindspore/ops/kernel/cpu/pyboost/customize/muls.h"
#include <memory>
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/mul.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MulsCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const ScalarPtr &other) {
  MS_LOG(DEBUG) << "Muls Call start";
  OpRunner::InferOpOutput(op, input_tensor, other);
  const auto &device_name = device::DeviceType::kCPU;

  // the Muls primitive does not support CPU, so use Mul instead.
  const auto mul_op = CREATE_PYBOOST_OP(Mul, device_name);

  // handle type promotion manually since the CPU kernelmod Mul does not support it
  const auto out_dtype = op->output(0)->Dtype();
  const auto other_tensor = PyBoostUtils::ScalarToTensor(other, out_dtype);
  const auto mul_out = mul_op->Call(input_tensor, other_tensor);
  op->set_outputs({mul_out});
  MS_LOG(DEBUG) << "Muls Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
