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
#include "kernel/ascend/aclnn/pyboost_impl/customize/nan_to_num.h"
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<float, float> GetInfValues(TypeId input_type, const std::optional<FP32ImmPtr> &posinf,
                                      const std::optional<FP32ImmPtr> &neginf, bool posinf_has_value,
                                      bool neginf_has_value) {
  const float DOUBLE_MAX_VALUE = 1.7976931348623157e+308;
  const float DOUBLE_MIN_VALUE = -1.7976931348623157e+308;
  const float FLOAT32_MAX_VALUE = 3.4028235e+38;
  const float FLOAT32_MIN_VALUE = -3.4028235e+38;
  const float FLOAT16_MAX_VALUE = 65504.0;
  const float FLOAT16_MIN_VALUE = -65504.0;
  const float BFLOAT16_MAX_VALUE = 3.3895314e+38;
  const float BFLOAT16_MIN_VALUE = -3.3895314e+38;
  float new_posinf{0.0};
  float new_neginf{0.0};
  switch (input_type) {
    case kNumberTypeFloat64:
      new_posinf = posinf_has_value ? posinf.value()->value() : DOUBLE_MAX_VALUE;
      new_neginf = neginf_has_value ? neginf.value()->value() : DOUBLE_MIN_VALUE;
      break;
    case kNumberTypeFloat32:
      new_posinf = posinf_has_value ? posinf.value()->value() : FLOAT32_MAX_VALUE;
      new_neginf = neginf_has_value ? neginf.value()->value() : FLOAT32_MIN_VALUE;
      break;
    case kNumberTypeFloat16:
      new_posinf = posinf_has_value ? posinf.value()->value() : FLOAT16_MAX_VALUE;
      new_neginf = neginf_has_value ? neginf.value()->value() : FLOAT16_MIN_VALUE;
      break;
    case kNumberTypeBFloat16:
      new_posinf = posinf_has_value ? posinf.value()->value() : BFLOAT16_MAX_VALUE;
      new_neginf = neginf_has_value ? neginf.value()->value() : BFLOAT16_MIN_VALUE;
      break;
    default:
      new_posinf = posinf_has_value ? posinf.value()->value() : FLOAT32_MAX_VALUE;
      new_neginf = neginf_has_value ? neginf.value()->value() : FLOAT32_MIN_VALUE;
      break;
  }
  return std::make_tuple(new_posinf, new_neginf);
}
tensor::TensorPtr NanToNumAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                          const std::optional<FP32ImmPtr> &nan, const std::optional<FP32ImmPtr> &posinf,
                                          const std::optional<FP32ImmPtr> &neginf) {
  const float DEFAULT_NAN = 0.0;

  auto new_nan = nan.has_value() ? nan.value()->value() : DEFAULT_NAN;
  OpRunner::InferOpOutput(op, input_tensor, nan, posinf, neginf);

  bool posinf_has_value = posinf.has_value();
  bool neginf_has_value = neginf.has_value();

  float new_posinf{0.0};
  float new_neginf{0.0};
  if (posinf_has_value && neginf_has_value) {
    new_posinf = posinf.value()->value();
    new_neginf = neginf.value()->value();
  } else {
    auto input_type = input_tensor->Dtype()->type_id();
    std::tie(new_posinf, new_neginf) = GetInfValues(input_type, posinf, neginf, posinf_has_value, neginf_has_value);
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, new_nan, new_posinf, new_neginf]() {
      MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
      auto device_context = op->device_context();

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());
      LAUNCH_ACLNN(aclnnNanToNum, device_context, op->stream_id(), input_tensor, new_nan, new_posinf, new_neginf,
                   op->output(0));

      MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
    }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
