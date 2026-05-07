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

#include "kernel/ascend/aclnn/pyboost_impl/customize/hshrink_grad.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr HShrinkGradAscendCustomize(const TensorPtr &gradients_tensor, const TensorPtr &features_tensor,
                                             const ScalarPtr &lambd) {
  auto output = InferOutput<1>(prim::kPrimHShrinkGrad, gradients_tensor, features_tensor, lambd);
  TensorToDevice(gradients_tensor, features_tensor);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnHardshrinkBackward, gradients_tensor, features_tensor, lambd, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
