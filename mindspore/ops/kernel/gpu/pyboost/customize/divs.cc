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

#include "kernel/gpu/pyboost/customize/divs.h"
#include "kernel/gpu/pyboost/auto_generate/div.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DivsGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const ScalarPtr &other) {
  MS_LOG(DEBUG) << "Divs Call start";
  OpRunner::InferOpOutput(op, input_tensor, other);

  const auto &device_name = device::DeviceType::kGPU;

  // the Divs primitive does not support GPU, so use Div instead.
  const auto div_op = CREATE_PYBOOST_OP(Div, device_name);

  // handle type promotion manually since the GPU kernelmod Div does not support it
  const auto out_dtype = op->output(0)->Dtype();
  auto input_tensor_cast = input_tensor;
  if (input_tensor->Dtype()->type_id() != out_dtype->type_id()) {
    input_tensor_cast = PyBoostUtils::CastTensor(input_tensor, out_dtype->type_id(), device_name);
  }
  const auto other_tensor = PyBoostUtils::ScalarToTensor(other, out_dtype);

  const auto div_out = div_op->Call(input_tensor_cast, other_tensor);
  op->set_outputs({div_out});

  MS_LOG(DEBUG) << "Divs Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
