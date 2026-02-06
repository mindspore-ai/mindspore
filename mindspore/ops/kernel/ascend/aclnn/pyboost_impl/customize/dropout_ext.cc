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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dropout_ext.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> DropoutExtAscendCustomize(const tensor::TensorPtr &input,
                                                                           const FP32ImmPtr &p,
                                                                           const tensor::TensorPtr &seed,
                                                                           const tensor::TensorPtr &offset) {
  auto outputs = InferOutput<2>(prim::kPrimDropoutExt, input, p, seed, offset);
  auto [output0, output1] = outputs;
  TensorToDevice(input);
  TensorAllocate(outputs);
  auto gen_state = GetGeneratorState(seed, offset);
  auto seed_value = std::get<0>(gen_state);
  auto offset_value = std::get<1>(gen_state);
  auto p_value = static_cast<double>(p->value());

  std::vector<int64_t> input_shape = input->shape();
  auto dtype = input->data_type();
  MS_LOG(DEBUG) << "DropoutExt input dtype = " << TypeIdToString(dtype);

  DISPATCH_ACLNN(aclnnDropoutGenMaskV2, input_shape, p_value, seed_value, offset_value, dtype, output1);
  DISPATCH_ACLNN(aclnnDropoutDoMask, input, output1, p_value, output0);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
