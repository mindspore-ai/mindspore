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

#include "kernel/ascend/aclnn/pyboost_impl/customize/divmods.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DivModsAscendCustomize(const TensorPtr &input, const ScalarPtr &other,
                                         const std::optional<Int64ImmPtr> &rounding_mode) {
  auto output = InferOutput<1>(prim::kPrimDivMods, input, other, rounding_mode);

  ScalarPtr other_real = ops::FetchRealScalar(other);
  int64_t mode = 0;
  if (rounding_mode.has_value()) {
    mode = GetValue<int64_t>(rounding_mode.value());
  }

  TensorToDevice(input);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnDivMods, input, other_real, mode, output);

  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
