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

#include "plugin/device/gpu/kernel/pyboost/customize/mean_ext.h"
#include "plugin/device/gpu/kernel/pyboost/auto_generate/cast.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void MeanExtGPUCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const ValuePtr &axis,
                    const BoolImmPtr &keep_dims) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis, keep_dims]() {
    MS_LOG(DEBUG) << "For 'MeanExt', the gpu task 'ReduceMean' start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    const auto primitive = std::make_shared<Primitive>(prim::kPrimReduceMean->name());
    MS_EXCEPTION_IF_NULL(primitive);

    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Set new Abstract for ReduceMean
    std::vector<AbstractBasePtr> input_abs{input_tensor->ToAbstract(), axis->ToAbstract(), keep_dims->ToAbstract()};

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
    MS_LOG(DEBUG) << "For 'MeanExt', the gpu task 'ReduceMean' end";
  }));
}
}  // namespace

void MeanExtGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                         const std::optional<ValueTuplePtr> &axis, const BoolImmPtr &keep_dims,
                         const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, input_tensor, axis, keep_dims, dtype);

  // If axis is None, Convert axis to empty tuple
  ValuePtr act_axis;
  if (axis.has_value()) {
    act_axis = axis.value();
  } else {
    act_axis = MakeValue<std::vector<int64_t>>({});
  }

  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = op->output_abs()->GetType()->cast<TensorTypePtr>()->element()->type_id();

  TensorPtr act_tensor = input_tensor;
  // Call Cast before Launch ReduceMean
  if (input_tensor->data_type() != out_dtype) {
    MS_LOG(DEBUG) << "Call Cast gpu kernel, src dtype: " << TypeIdToString(input_tensor->data_type())
                  << ", dst dtype: " << TypeIdToString(out_dtype);
    const auto &cast_op = CREATE_PYBOOST_OP(Cast, op->device_context()->device_context_key_.device_name_);
    cast_op->set_primitive(prim::kPrimCast);
    act_tensor = cast_op->Call(input_tensor, std::make_shared<Int64Imm>(out_dtype));
  }

  MeanExtGPUCall(op, act_tensor, act_axis, keep_dims);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
