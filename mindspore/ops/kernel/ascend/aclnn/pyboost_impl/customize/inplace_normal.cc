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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_normal.h"
#include <memory>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
float GetScalarValueToFloat(const std::shared_ptr<Scalar> &scalar, const string &scalar_name) {
  if (scalar->isa<Int32Imm>()) {
    return static_cast<float>(GetValue<int32_t>(scalar));
  } else if (scalar->isa<Int64Imm>()) {
    return static_cast<float>(GetValue<int64_t>(scalar));
  } else if (scalar->isa<FP64Imm>()) {
    return static_cast<float>(GetValue<double>(scalar));
  } else if (scalar->isa<FP32Imm>()) {
    return GetValue<float>(scalar);
  } else {
    MS_EXCEPTION(TypeError) << "For Pyboost InplaceNormal, the input " << scalar_name
                            << " does not support dtype: " << scalar->type_name();
  }
}

tensor::TensorPtr InplaceNormalAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                               const ScalarPtr &mean, const ScalarPtr &std, const TensorPtr &seed,
                                               const TensorPtr &offset) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  op->set_outputs({input});

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, mean, std, seed, offset]() {
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input);
    // For the aclnnInplacenormal operator, mean and std must be of the float type.
    float mean_value = GetScalarValueToFloat(mean, "mean");
    float std_value = GetScalarValueToFloat(std, "std");
    auto [seed_value, offset_value] = GetGeneratorState(seed, offset);

    MS_LOG(DEBUG) << "InplaceNormal launch start";
    LAUNCH_ACLNN(aclnnInplaceNormal, device_context, op->stream_id(), input, mean_value, std_value, seed_value,
                 offset_value);
    MS_LOG(DEBUG) << "Run device task aclnnInplaceNormal end";
  }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
