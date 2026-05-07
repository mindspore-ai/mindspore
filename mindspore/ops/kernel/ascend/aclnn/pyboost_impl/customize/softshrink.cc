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

#include "kernel/ascend/aclnn/pyboost_impl/customize/softshrink.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SoftShrinkAscendCustomize(const TensorPtr &input_tensor, const ScalarPtr &lambd) {
  MS_LOG(DEBUG) << "Softshrink Ascend start";
  auto output = InferOutput<1>(prim::kPrimSoftShrink, input_tensor, lambd);
  TensorToDevice(input_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnSoftshrink, input_tensor, lambd, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
