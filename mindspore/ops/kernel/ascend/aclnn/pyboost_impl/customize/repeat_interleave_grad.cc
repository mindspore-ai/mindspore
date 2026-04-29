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

#include "kernel/ascend/aclnn/pyboost_impl/customize/repeat_interleave_grad.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RepeatInterleaveGradAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                      const tensor::TensorPtr &repeats, const Int64ImmPtr &dim) {
  auto output = InferOutput<1>(prim::kPrimRepeatInterleaveGrad, input_tensor, repeats, dim);
  const ShapeVector &output_shape = output->shape();
  auto repeats_tensor = repeats;
  auto repeats_shape = repeats_tensor->shape();
  if (repeats_shape.empty()) {
    repeats_tensor = std::make_shared<Tensor>(*repeats);
    repeats_tensor->set_shape(ShapeVector{1});
  }

  TensorToDevice(input_tensor, repeats_tensor);
  TensorAllocate(output);

  int64_t dim_imm = GetValue<int64_t>(dim);
  auto rank = SizeToLong(output_shape.size());
  dim_imm = (dim_imm < 0) ? (dim_imm + rank) : dim_imm;
  MS_LOG(DEBUG) << "RepeatInterleaveGrad Call start";

  // Async
  DISPATCH_ACLNN(aclnnRepeatInterleaveGrad, input_tensor, repeats_tensor, dim_imm, output);

  MS_LOG(DEBUG) << "RepeatInterleaveGrad Launch end";
  return output;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
