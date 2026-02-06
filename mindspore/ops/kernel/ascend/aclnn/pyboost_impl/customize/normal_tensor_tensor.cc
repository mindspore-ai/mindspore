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

#include "kernel/ascend/aclnn/pyboost_impl/customize/normal_tensor_tensor.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NormalTensorTensorAscendCustomize(const TensorPtr &mean_tensor, const TensorPtr &std_tensor,
                                                    const TensorPtr &seed, const TensorPtr &offset) {
  MS_LOG(DEBUG) << "NormalTensorTensor call start";
  auto output = InferOutput<1>(prim::kPrimNormalTensorTensor, mean_tensor, std_tensor, seed, offset);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);

  TensorToDevice(mean_tensor, std_tensor);
  TensorAllocate(output);

  DISPATCH_ACLNN(aclnnNormalTensorTensor, mean_tensor, std_tensor, seed_imm, offset_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
