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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_bernoulli_tensor.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceBernoulliTensorAscendCustomize(const TensorPtr &input, const TensorPtr &p,
                                                        const TensorPtr &seed, const TensorPtr &offset) {
  MS_LOG(DEBUG) << "Call InplaceBernoulliTensor start";
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  TensorToDevice(input, p);
  DISPATCH_ACLNN(aclnnInplaceBernoulliTensor, input, p, seed_imm, offset_imm);
  return input;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
