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

#include "mindspore/ops/kernel/cpu/pyboost/customize/sum_ext.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/cast.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/sum_ext.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SumExtCPUCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const ValuePtr &axis,
                   const BoolImmPtr &keep_dims, const BoolImmPtr &skip_mode,
                   const std::vector<AbstractBasePtr> &input_abs) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, axis, keep_dims, skip_mode, input_abs]() {
      MS_LOG(DEBUG) << "For 'SumExt', the cpu task 'ReduceSum' start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      const auto primitive = std::make_shared<Primitive>(prim::kPrimReduceSum->name());
      MS_EXCEPTION_IF_NULL(primitive);

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs,
                                                                    input_tensor, axis, keep_dims, skip_mode);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info);
      MS_LOG(DEBUG) << "For 'SumExt', the cpu task 'ReduceSum' end";
    }));
}
}  // namespace

void SumExtCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
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
  TypeId out_dtype = op->output_value_simple_info()->dtype_vector_[kIndex0]->type_id();

  TensorPtr act_tensor = input_tensor;
  // Call Cast before Launch ReduceSum
  if (input_tensor->data_type() != out_dtype) {
    MS_LOG(DEBUG) << "Call Cast cpu kernel, src dtype: " << TypeIdToString(input_tensor->data_type())
                  << ", dst dtype: " << TypeIdToString(out_dtype);
    act_tensor = PyBoostUtils::CastTensor(input_tensor, out_dtype, device::DeviceType::kCPU);
  }

  if (act_tensor->data_type() == kNumberTypeFloat16) {
    // Increase the precision to float32 for calculation
    const auto &cast_input_tensor = PyBoostUtils::CastTensor(act_tensor, kNumberTypeFloat32, device::DeviceType::kCPU);
    const auto &sum_ext_op = CREATE_PYBOOST_OP(SumExt, device::DeviceType::kCPU);
    const auto &cast_output_tensor = sum_ext_op->Call(cast_input_tensor, axis, keep_dims, std::nullopt);
    // After calculation, reduce the precision to float16
    const auto &output_tensor =
      PyBoostUtils::CastTensor(cast_output_tensor, kNumberTypeFloat16, device::DeviceType::kCPU);
    op->set_outputs({output_tensor});
  } else {
    const auto skip_mode = std::make_shared<BoolImm>(false);
    // Set new input abstract for ReduceSum
    std::vector<AbstractBasePtr> new_input_abs{act_tensor->ToAbstract(), act_axis->ToAbstract(),
                                               keep_dims->ToAbstract(), skip_mode->ToAbstract()};
    SumExtCPUCall(op, act_tensor, act_axis, keep_dims, skip_mode, new_input_abs);
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
