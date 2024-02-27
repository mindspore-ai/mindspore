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

#include "plugin/device/gpu/kernel/pyboost/customize/min.h"
#include "plugin/device/gpu/kernel/pyboost/customize/max.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void MinOrMaxGPUCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const std::string &reduce_op) {
  MS_EXCEPTION_IF_NULL(op);
  OpRunner::InferOpOutput(op, input_tensor);
  auto axis = MakeValue<std::vector<int64_t>>({});
  auto keep_dims = MakeValue<bool>(false);
  std::vector<AbstractBasePtr> input_abs{input_tensor->ToAbstract(), axis->ToAbstract(), keep_dims->ToAbstract()};

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis, keep_dims, input_abs, reduce_op]() {
      MS_LOG(DEBUG) << "For '" << op->primitive()->name() << "', the gpu task start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      const auto primitive = std::make_shared<Primitive>(reduce_op);
      MS_EXCEPTION_IF_NULL(primitive);

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      const auto &input_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, input_tensor, axis, keep_dims);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);
      auto stream = device::gpu::GPUDeviceManager::GetInstance().GetStream(op->stream_id());
      PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info, stream);

      static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
      if (sync && !device_context->device_res_manager_->SyncAllStreams()) {
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << primitive->name();
      }
      MS_LOG(DEBUG) << "For '" << op->primitive()->name() << "', the gpu task end";
    }));
}
}  // namespace

void MinGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MinOrMaxGPUCall(op, input_tensor, prim::kPrimReduceMin->name());
}

void MaxGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MinOrMaxGPUCall(op, input_tensor, prim::kPrimReduceMax->name());
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
