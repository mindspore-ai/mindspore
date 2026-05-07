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

#include "kernel/ascend/aclnn/pyboost_impl/customize/floor_div.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr FloorDivAscendCustomize(const tensor::TensorPtr &input, const tensor::TensorPtr &other) {
  auto output = InferOutput<1>(prim::kPrimFloorDiv, input, other);

  TensorToDevice(input, other);
  TensorMalloc(output);

  // Async
  DISPATCH_ACLNN(aclnnFloorDivide, input, other, output);
  MS_LOG(DEBUG) << "Run device task FloorDiv end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
