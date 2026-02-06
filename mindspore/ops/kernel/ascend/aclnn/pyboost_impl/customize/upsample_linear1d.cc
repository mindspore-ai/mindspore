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

#include "kernel/ascend/aclnn/pyboost_impl/customize/upsample_linear1d.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr UpsampleLinear1DAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                  const std::optional<ValueTuplePtr> &output_size,
                                                  const std::optional<ValueTuplePtr> &scale_factors,
                                                  const BoolImmPtr &align_corners) {
  MS_LOG(INFO) << "UpsampleLinear1DAscendCustomize start";

  auto output = InferOutput<1>(prim::kPrimUpsampleLinear1D, input_tensor, output_size, scale_factors, align_corners);

  // get output_size, scale_factors and align_corners
  const ShapeVector &osize = output->shape();
  std::vector<int64_t> output_size_vector = {osize.begin() + kDim2, osize.end()};
  auto align_corners_val = GetValue<bool>(align_corners);

  const pyfloat DEFAULT_SCALE_VALUE = -1;
  std::vector<pyfloat> scales{DEFAULT_SCALE_VALUE};
  if (scale_factors.has_value()) {
    scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
  }

  TensorToDevice(input_tensor);
  TensorAllocate(output);
  const double UpsampleLinear1dEps = 1e-7;
  double scale_l = scales.at(0) != DEFAULT_SCALE_VALUE ? (static_cast<double>(scales.at(0)) + UpsampleLinear1dEps)
                                                       : DEFAULT_SCALE_VALUE;
  DISPATCH_ACLNN(aclnnUpsampleLinear1d, input_tensor, output_size_vector, align_corners_val, scale_l, output);

  MS_LOG(INFO) << "UpsampleLinear1DAscendCustomize end";

  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
