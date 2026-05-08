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
 * limitations under the License.∏
 */

#include "kernel/ascend/aclnn/pyboost_impl/customize/gelu_grad_ext.h"

#include <string>
#include <unordered_map>

#include "ir/scalar.h"
#include "include/op_enum.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
using ops::Approximate;
const std::unordered_map<Approximate, std::string> ApproximateModeMap{{Approximate::NONE, "none"},
                                                                      {Approximate::TANH, "tanh"}};
std::string GetApproximateMode(int64_t approximate) {
  auto approximate_enum = static_cast<Approximate>(approximate);
  auto it = ApproximateModeMap.find(approximate_enum);
  if (it == ApproximateModeMap.end()) {
    MS_EXCEPTION(ValueError) << "The value of approximate should be 0 or 1, but got " << approximate;
  }
  return it->second;
}
}  // namespace
tensor::TensorPtr GeluGradExtAscendCustomize(const TensorPtr &grad_tensor, const TensorPtr &input_tensor,
                                             const Int64ImmPtr &approximate) {
  auto output = InferOutput<1>(prim::kPrimGeluGradExt, grad_tensor, input_tensor, approximate);

  auto approximate_value = GetValue<int64_t>(approximate);
  auto mode = GetApproximateMode(approximate_value);

  // Create device address for input/output tensors
  TensorToDevice(grad_tensor, input_tensor);
  TensorMalloc(output);
  MS_LOG(DEBUG) << "GeluGradExt Call start";
  DISPATCH_ACLNN(aclnnGeluBackwardV2, grad_tensor, input_tensor, mode, output);
  MS_LOG(DEBUG) << "GeluGradExt Launch end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
