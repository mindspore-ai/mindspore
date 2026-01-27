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

#include <vector>
#include "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/view_impl.h"
#include "view/reshape_strides_calc.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "mindspore/core/include/utils/stream_guard.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_reg.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/pynative/utils/pyboost/functions/dispatch.h"

namespace mindspore::kernel::pyboost {
inline device::DeviceType GetDeviceTarget() { return OpRunStatus::Get().device_target(); }
mindspore::tensor::TensorPtr reshape_impl(const mindspore::tensor::TensorPtr &input,
                                          const std::vector<int64_t> &shape) {
  auto storage_info = ops::ReshapeBasicTypeCalc(input, shape);

  device::DeviceType device_target;
  if (EnableDispatch()) {
    device_target = get_device(input);
  } else {
    device_target = GetDeviceTarget();
  }

  if (MS_LIKELY(storage_info)) {
    OpRunStatus::Get().HeterBarrier(device_target);
    MS_LOG(DEBUG) << "View Reshape Call start";
    tensor::TensorPtrList outputs;
    // device info
    const auto &device_context = runtime::OpRunner::GetDeviceContext(device_target);
    auto cur_stream_id = CurrentStream::id();

    kernel::pyboost::PyBoostUtils::PrepareOpInputs(device_context, cur_stream_id, input);
    kernel::pyboost::PyBoostUtils::CreateOutputTensor(device_context, input, storage_info, &outputs);

    // Async
    kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([input, device_context]() {
      MS_LOG(DEBUG) << "View device task Reshape start";
      kernel::pyboost::PyBoostUtils::MallocOpInputs(device_context, input);
      MS_LOG(DEBUG) << "View device task Reshape end";
    }));

    static auto reshape_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().ReshapeGradFuncObj;
    reshape_grad_func(outputs[0], input, shape);
    MS_LOG(DEBUG) << "View Reshape Call end";
    return outputs[0];
  }

  MS_LOG(DEBUG) << "View Contiguous + Unsafe View Call start";
  const auto contig_tensor = contiguous(input);
  IsSafeViewGuard safe_view_guard(false);
  auto output = view(contig_tensor, shape);
  MS_LOG(DEBUG) << "View Contiguous + Unsafe View Call end";
  return output;
}
}  // namespace mindspore::kernel::pyboost
