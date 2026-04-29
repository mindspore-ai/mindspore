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

#include "kernel/ascend/aclnn/pyboost_impl/customize/multinomial_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MultinomialExtAscendCustomize(const tensor::TensorPtr &input_tensor, const Int64ImmPtr &num_samples,
                                                const BoolImmPtr &replacement, const tensor::TensorPtr &seed,
                                                const tensor::TensorPtr &offset) {
  MS_LOG(DEBUG) << prim::kPrimMultinomialExt->name() << " call start";
  auto output = InferOutput<1>(prim::kPrimMultinomialExt, input_tensor, num_samples, replacement, seed, offset);

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  TensorToDevice(input_tensor);
  TensorAllocate(output);

  auto num_samples_value = static_cast<int64_t>(num_samples->value());
  auto replacement_value = static_cast<bool>(replacement->value());
  DISPATCH_ACLNN(aclnnMultinomial, input_tensor, num_samples_value, replacement_value, seed_imm, offset_imm, output);
  MS_LOG(DEBUG) << "Run device task " << prim::kPrimMultinomialExt->name() << " end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
