/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/avg_pool3d_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AvgPool3DExtAscendCustomize(const TensorPtr &input, const ValueTuplePtr &kernel_size,
                                              const std::optional<mindspore::ValueTuplePtr> &stride,
                                              const ValueTuplePtr &padding, const BoolImmPtr &ceil_mode,
                                              const BoolImmPtr &count_include_pad,
                                              const std::optional<Int64ImmPtr> &divisor_override) {
  MS_LOG(INFO) << "AvgPool3DExtAscendCustomize start";
  auto output = InferOutput<1>(prim::kPrimAvgPool3DExt, input, kernel_size, stride, padding, ceil_mode,
                               count_include_pad, divisor_override);

  auto kernel_size_val = ConvertValueTupleToVector<int64_t>(kernel_size);
  auto stride_val = stride.has_value() ? ConvertValueTupleToVector<int64_t>(stride.value()) : kernel_size_val;
  auto padding_val = ConvertValueTupleToVector<int64_t>(padding);
  auto ceil_mode_val = GetValue<bool>(ceil_mode);
  auto count_include_pad_val = GetValue<bool>(count_include_pad);
  auto divisor_override_val = divisor_override.has_value() ? GetValue<int64_t>(divisor_override.value()) : 0;

  TensorToDevice(input);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnAvgPool3d, input, kernel_size_val, stride_val, padding_val, ceil_mode_val, count_include_pad_val,
                 divisor_override_val, output);

  MS_LOG(INFO) << "AvgPool3DExtAscendCustomize end";

  return output;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
