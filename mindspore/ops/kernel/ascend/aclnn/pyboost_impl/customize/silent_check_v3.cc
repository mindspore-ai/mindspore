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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/silent_check_v3.h"
#include <cassert>
#include <vector>
#include "mindapi/base/shape_vector.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "mindapi/base/types.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> SilentCheckV3AscendCustomize(
  const TensorPtr &val, const TensorPtr &max, const TensorPtr &avg, const TensorPtr &input_grad, const TensorPtr &step,
  const FloatImmPtr &c_thresh_l1, const FloatImmPtr &c_thresh_l2, const FloatImmPtr &beta1,
  const Int64ImmPtr &npu_asd_detect) {
  MS_LOG(INFO) << "SilentCheckV3 Call start";
  auto outputs = InferOutput<4>(prim::kPrimSilentCheckV3, val, max, avg, input_grad, step, c_thresh_l1, c_thresh_l2,
                                beta1, npu_asd_detect);

  auto c_thresh_l1_value = GetValue<pyfloat>(c_thresh_l1);
  auto c_thresh_l2_value = GetValue<pyfloat>(c_thresh_l2);
  auto beta1_value = GetValue<pyfloat>(beta1);
  auto npu_asd_detect_value = GetValue<int64_t>(npu_asd_detect);

  TensorToDevice(val, max, avg, input_grad, step);
  const auto &result = std::get<3>(outputs);
  TensorAllocate(result);
  DISPATCH_ACLNN(aclnnSilentCheckV2, val, max, avg, input_grad, step, input_grad->shape_c(), input_grad->stride(),
                 ShapeVector({SizeToLong(input_grad->storage_offset())}), c_thresh_l1_value, c_thresh_l2_value,
                 beta1_value, npu_asd_detect_value, result);

  MS_LOG(INFO) << "SilentCheckV3 Launch end";
  return std::make_tuple(avg, input_grad, step, result);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
