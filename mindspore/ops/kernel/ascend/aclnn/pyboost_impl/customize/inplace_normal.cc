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
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"

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

tensor::TensorPtr InplaceNormalAscendCustomize(const tensor::TensorPtr &input, const ScalarPtr &mean,
                                               const ScalarPtr &std, const tensor::TensorPtr &seed,
                                               const tensor::TensorPtr &offset) {
  float mean_value = GetScalarValueToFloat(mean, "mean");
  float std_value = GetScalarValueToFloat(std, "std");
  auto gen_state = GetGeneratorState(seed, offset);
  auto seed_value = std::get<0>(gen_state);
  auto offset_value = std::get<1>(gen_state);
  TensorToDevice(input);
  DISPATCH_ACLNN(aclnnInplaceNormal, input, mean_value, std_value, seed_value, offset_value);
  return input;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
