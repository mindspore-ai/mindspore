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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_random.h"

#include <limits>

#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "ops_utils/type_dispatch.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
template <typename T>
void GetTypeLimit(int64_t *limit) {
  if constexpr (std::is_floating_point_v<T>) {
    constexpr int64_t base = 1;
    *limit = (base << std::numeric_limits<T>::digits) + 1;
  } else {
    *limit = std::numeric_limits<T>::max();
  }
}
}  // namespace

tensor::TensorPtr InplaceRandomAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor_tensor,
                                               const Int64ImmPtr from, const std::optional<Int64ImmPtr> &to,
                                               const TensorPtr &seed, const TensorPtr &offset) {
  OpRunner::InferOpOutput(op, tensor_tensor, from, to, seed, offset);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  auto from_imm = GetValueWithCheck<int64_t>(from);
  int64_t to_imm;
  if (to.has_value()) {
    to_imm = GetValueWithCheck<int64_t>(to.value());
  } else {
    TYPE_DISPATCH_ALL(tensor_tensor->Dtype()->type_id(), "GetTypeLimit",
                      [&]() -> void { GetTypeLimit<scalar_t>(&to_imm); });
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensor_tensor);
  op->set_outputs({tensor_tensor});

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, tensor_tensor, seed_imm, offset_imm, from_imm, to_imm]() {
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, tensor_tensor);
      LAUNCH_ACLNN(aclnnInplaceRandom, device_context, op->stream_id(), tensor_tensor, from_imm, to_imm, seed_imm,
                   offset_imm);
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
