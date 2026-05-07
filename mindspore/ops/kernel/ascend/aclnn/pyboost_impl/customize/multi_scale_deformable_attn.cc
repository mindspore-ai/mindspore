/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/multi_scale_deformable_attn.h"
#include <tuple>
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/cast.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MultiScaleDeformableAttnAscendCustomize(const TensorPtr &value_tensor, const TensorPtr &shape_tensor,
                                                          const TensorPtr &offset_tensor,
                                                          const TensorPtr &locations_tensor,
                                                          const TensorPtr &weight_tensor) {
  auto ori_type = value_tensor->Dtype();

  auto type = value_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, value tensor type " << type << " is illegal";
  }

  type = shape_tensor->Dtype();
  if (type != kInt32 && type != kInt64) {
    MS_LOG(EXCEPTION) << "For MSDA, shape tensor type " << type << " is illegal";
  }

  type = offset_tensor->Dtype();
  if (type != kInt32 && type != kInt64) {
    MS_LOG(EXCEPTION) << "For MSDA, offset tensor type " << type << " is illegal";
  }

  type = locations_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, locations tensor type " << type << " is illegal";
  }

  type = weight_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, weight tensor type " << type << " is illegal";
  }

  auto value_tensor_cp = PyBoostUtils::CastTensor(value_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);
  auto shape_tensor_cp = PyBoostUtils::CastTensor(shape_tensor, kNumberTypeInt32, device::DeviceType::kAscend);
  auto offset_tensor_cp = PyBoostUtils::CastTensor(offset_tensor, kNumberTypeInt32, device::DeviceType::kAscend);
  auto locations_tensor_cp =
    PyBoostUtils::CastTensor(locations_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);
  auto weight_tensor_cp = PyBoostUtils::CastTensor(weight_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);

  auto output = InferOutput<1>(prim::kPrimMultiScaleDeformableAttn, value_tensor_cp, shape_tensor_cp, offset_tensor_cp,
                               locations_tensor_cp, weight_tensor_cp);

  TensorToDevice(value_tensor_cp, shape_tensor_cp, offset_tensor_cp, locations_tensor_cp, weight_tensor_cp);
  TensorMalloc(output);

  // Async
  DISPATCH_ACLNN(aclnnMultiScaleDeformableAttnFunction, value_tensor_cp, shape_tensor_cp, offset_tensor_cp,
                 locations_tensor_cp, weight_tensor_cp, output);

  if (ori_type == kFloat32) {
    return PyBoostUtils::CastTensor(output, kNumberTypeFloat32, device::DeviceType::kAscend);
  }
  return PyBoostUtils::CastTensor(output, kNumberTypeFloat16, device::DeviceType::kAscend);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
