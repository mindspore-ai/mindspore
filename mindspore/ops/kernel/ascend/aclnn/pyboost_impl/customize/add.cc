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

#include "kernel/ascend/aclnn/pyboost_impl/customize/add.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AddAscendCustomize(const TensorPtr &x_tensor, const TensorPtr &y_tensor) {
  auto output = InferOutput<1>(prim::kPrimAdd, x_tensor, y_tensor);
  // No need to convert input
  TensorToDevice(x_tensor, y_tensor);
  TensorMalloc(output);
  // Async
  const ScalarPtr kAddAlpha = std::make_shared<Int64Imm>(1);
  DISPATCH_ACLNN(aclnnAdd, x_tensor, y_tensor, kAddAlpha, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
