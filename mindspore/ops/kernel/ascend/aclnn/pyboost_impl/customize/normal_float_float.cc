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

#include "kernel/ascend/aclnn/pyboost_impl/customize/normal_float_float.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NormalFloatFloatAscendCustomize(const ScalarPtr &mean_float, const ScalarPtr &std_float,
                                                  const ValueTuplePtr &size, const TensorPtr &seed,
                                                  const TensorPtr &offset) {
  MS_LOG(DEBUG) << "NormalFloatFloat call start";
  auto output = InferOutput<1>(prim::kPrimNormalFloatFloat, mean_float, std_float, size, seed, offset);
  auto mean_imm = GetScalarCastValue<float>("NormalFloatFloat", mean_float);
  auto std_imm = GetScalarCastValue<float>("NormalFloatFloat", std_float);
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);

  TensorAllocate(output);

  DISPATCH_ACLNN(aclnnNormalFloatFloat, mean_imm, std_imm, seed_imm, offset_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
