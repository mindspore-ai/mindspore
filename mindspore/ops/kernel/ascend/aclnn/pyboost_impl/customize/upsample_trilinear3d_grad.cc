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

#include "kernel/ascend/aclnn/pyboost_impl/customize/upsample_trilinear3d_grad.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr UpsampleTrilinear3DGradAscendCustomize(const tensor::TensorPtr &gradout_tensor,
                                                         const ValueTuplePtr &input_size,
                                                         const std::optional<ValueTuplePtr> &output_size,
                                                         const std::optional<ValueTuplePtr> &scale_factors,
                                                         const BoolImmPtr &align_corners) {
  MS_LOG(DEBUG) << "UpsampleTrilinear3DGradAscendCustomize start";
  auto output = InferOutput<1>(prim::kPrimUpsampleTrilinear3DGrad, gradout_tensor, input_size, output_size,
                               scale_factors, align_corners);

  auto input_size_vector = ConvertValueTupleToVector<int64_t>(input_size);

  std::vector<int64_t> output_size_vector{};
  constexpr pyfloat DEFAULT_SCALE_VALUE = 0.;
  double scales_d = DEFAULT_SCALE_VALUE;
  double scales_h = DEFAULT_SCALE_VALUE;
  double scales_w = DEFAULT_SCALE_VALUE;
  if (output_size.has_value()) {
    output_size_vector = ConvertValueTupleToVector<int64_t>(output_size.value());
  } else if (scale_factors.has_value()) {
    auto scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
    for (size_t i = 0; i < scales.size(); ++i) {
      output_size_vector.push_back(static_cast<int64_t>(input_size_vector[i + kDim2]) * scales[i]);
    }
    scales_d = scales[0];
    scales_h = scales[1];
    scales_w = scales[2];
  }

  auto align_corners_val = GetValue<bool>(align_corners);

  TensorToDevice(gradout_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnUpsampleTrilinear3dBackward, gradout_tensor, output_size_vector, input_size_vector,
                 align_corners_val, scales_d, scales_h, scales_w, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
