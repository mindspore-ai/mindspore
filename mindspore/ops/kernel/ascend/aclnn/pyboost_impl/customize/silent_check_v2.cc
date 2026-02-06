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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/silent_check_v2.h"
#include <cassert>
#include <vector>
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
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> SilentCheckV2AscendCustomize(
  const TensorPtr &val, const TensorPtr &input_grad, const TensorPtr &sfda, const TensorPtr &step,
  const Int64ImmPtr &c_min_steps_ptr, const FloatImmPtr &c_thresh_l1_ptr, const FloatImmPtr &c_coeff_l1_ptr,
  const FloatImmPtr &c_thresh_l2_ptr, const FloatImmPtr &c_coeff_l2_ptr, const Int64ImmPtr &npu_asd_detect_ptr) {
  MS_LOG(INFO) << "SilentCheckV2 Call start";
  auto outputs = InferOutput<4>(prim::kPrimSilentCheckV2, val, input_grad, sfda, step, c_min_steps_ptr, c_thresh_l1_ptr,
                                c_coeff_l1_ptr, c_thresh_l2_ptr, c_coeff_l2_ptr, npu_asd_detect_ptr);

  auto c_min_steps = GetValue<int64_t>(c_min_steps_ptr);
  auto c_thresh_l1 = GetValue<pyfloat>(c_thresh_l1_ptr);
  auto c_coeff_l1 = GetValue<pyfloat>(c_coeff_l1_ptr);
  auto c_thresh_l2 = GetValue<pyfloat>(c_thresh_l2_ptr);
  auto c_coeff_l2 = GetValue<pyfloat>(c_coeff_l2_ptr);
  auto npu_asd_detect = GetValue<int64_t>(npu_asd_detect_ptr);

  TensorToDevice(val, input_grad, sfda, step);
  const auto &result = std::get<3>(outputs);
  TensorAllocate(result);
  DISPATCH_ACLNN(aclnnSilentCheck, val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2,
                 c_coeff_l2, npu_asd_detect, result);
  MS_LOG(INFO) << "SilentCheckV2 Launch end";
  return std::make_tuple(input_grad, sfda, step, result);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
