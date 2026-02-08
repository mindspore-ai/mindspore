/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/isinf.h"

#include <limits>

#include "include/pynative/utils/pyboost/auto_generate/abs.h"
#include "include/pynative/utils/pyboost/auto_generate/equal.h"

#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/core/include/base/float16.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IsInfAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "IsInfCustomize start";
  OpRunner::InferOpOutput(op, input_tensor);

  auto input_type = input_tensor->Dtype()->type_id();
  bool is_int_type = (input_type >= kNumberTypeBool) && (input_type < kNumberTypeFloat);

  TensorPtr abs_out = input_tensor;
  if (!is_int_type) {
    const auto abs_op = CREATE_PYBOOST_OP(Abs, device::DeviceType::kAscend);
    abs_out = abs_op->Call(input_tensor);
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), abs_out);
  }

  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, abs_out, is_int_type, input_type]() {
      MS_LOG(DEBUG) << "Run device task IsInf start";

      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      if (is_int_type) {
        LAUNCH_ACLNN(aclnnInplaceZero, op->device_context(), op->stream_id(), op->output(0));
        return;
      }

      PyBoostUtils::MallocOpInputs(device_context, abs_out);

      switch (input_type) {
        case kNumberTypeFloat:
        case kNumberTypeFloat32: {
          static const ScalarPtr inf_value = std::make_shared<FP32Imm>(std::numeric_limits<float>::infinity());
          LAUNCH_ACLNN(aclnnEqScalar, device_context, op->stream_id(), abs_out, inf_value, outputs[0]);
          break;
        }
        case kNumberTypeFloat64: {
          static const ScalarPtr inf_value = std::make_shared<FP64Imm>(std::numeric_limits<double>::infinity());
          LAUNCH_ACLNN(aclnnEqScalar, device_context, op->stream_id(), abs_out, inf_value, outputs[0]);
          break;
        }
        case kNumberTypeBFloat16: {
          static const ScalarPtr inf_value = std::make_shared<BF16Imm>(std::numeric_limits<BFloat16>::infinity());
          LAUNCH_ACLNN(aclnnEqScalar, device_context, op->stream_id(), abs_out, inf_value, outputs[0]);
          break;
        }
        case kNumberTypeFloat16: {
          static const ScalarPtr inf_value =
            std::make_shared<FP32Imm>(static_cast<float>(std::numeric_limits<float>::infinity()));
          LAUNCH_ACLNN(aclnnEqScalar, device_context, op->stream_id(), abs_out, inf_value, outputs[0]);
          break;
        }
        default:
          MS_EXCEPTION(TypeError) << "Op IsInf does not support type " << TypeIdToString(input_type);
          break;
      }
      MS_LOG(DEBUG) << "Run device task IsInf end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
