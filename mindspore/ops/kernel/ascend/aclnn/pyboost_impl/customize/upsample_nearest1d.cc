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

#include "kernel/ascend/aclnn/pyboost_impl/customize/upsample_nearest1d.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr UpsampleNearest1DAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                   const std::optional<ValueTuplePtr> &output_size,
                                                   const std::optional<ValueTuplePtr> &scale_factors) {
  auto output = InferOutput<1>(prim::kPrimUpsampleNearest1D, input_tensor, output_size, scale_factors);

  const ShapeVector &osize = output->shape();
  std::vector<int64_t> output_size_vector = {osize.begin() + kDim2, osize.end()};

  TensorToDevice(input_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnUpsampleNearest1d, input_tensor, output_size_vector, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
