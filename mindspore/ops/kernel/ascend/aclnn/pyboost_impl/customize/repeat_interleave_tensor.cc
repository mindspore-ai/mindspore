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

#include "kernel/ascend/aclnn/pyboost_impl/customize/repeat_interleave_tensor.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RepeatInterleaveTensorAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                        const tensor::TensorPtr &repeats,
                                                        const std::optional<Int64ImmPtr> &dim,
                                                        const std::optional<Int64ImmPtr> &output_size) {
  auto output = InferOutput<1>(prim::kPrimRepeatInterleaveTensor, input_tensor, repeats, dim, output_size);
  const ShapeVector &output_shape = output->shape();

  TensorToDevice(input_tensor, repeats);
  TensorMalloc(output);
  MS_LOG(DEBUG) << "RepeatInterleaveTensor Call start";

  if (dim.has_value()) {
    int64_t dim_imm = GetValue<int64_t>(*dim);
    auto rank = SizeToLong(output_shape.size());
    dim_imm = (dim_imm < 0) ? (dim_imm + rank) : dim_imm;
    if (output_size.has_value()) {
      int64_t output_size_imm = GetValue<int64_t>(*output_size);
      if (output_size_imm != output_shape[dim_imm]) {
        MS_EXCEPTION(RuntimeError) << "Allocated size does not match required size.";
      }
    }

    DISPATCH_ACLNN(aclnnRepeatInterleaveWithDim, input_tensor, repeats, dim_imm, output_shape[dim_imm], output);
  } else {
    if (output_size.has_value()) {
      int64_t output_size_imm = GetValue<int64_t>(*output_size);
      if (output_size_imm != output_shape[0]) {
        MS_EXCEPTION(RuntimeError) << "Allocated size does not match required size.";
      }
    }

    DISPATCH_ACLNN(aclnnRepeatInterleave, input_tensor, repeats, output_shape[0], output);
  }

  MS_LOG(DEBUG) << "RepeatInterleaveTensor Launch end";
  return output;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
