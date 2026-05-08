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

#include "kernel/ascend/aclnn/pyboost_impl/customize/type_as.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr TypeAsAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &other_tensor) {
  MS_LOG(DEBUG) << "Call type_as start";
  auto output = InferOutput<1>(prim::kPrimTypeAs, input_tensor, other_tensor);
  TensorToDevice(input_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnInplaceCopy, output, input_tensor);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
