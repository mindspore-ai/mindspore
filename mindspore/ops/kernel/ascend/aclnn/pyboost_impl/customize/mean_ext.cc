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

#include "kernel/ascend/aclnn/pyboost_impl/customize/mean_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MeanExtAscendCustomize(const TensorPtr &input_tensor, const std::optional<ValueTuplePtr> &axis,
                                         const BoolImmPtr &keep_dims, const std::optional<Int64ImmPtr> &dtype) {
  auto output = InferOutput<1>(prim::kPrimMeanExt, input_tensor, axis, keep_dims, dtype);

  std::vector<int64_t> axis_vector{};
  if (axis.has_value()) {
    axis_vector = ConvertValueTupleToVector<int64_t>(axis.value());
  }
  const auto keep_dims_imm = GetValue<bool>(keep_dims);

  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = output->data_type();

  TensorToDevice(input_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnMean, input_tensor, axis_vector, keep_dims_imm, out_dtype, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
