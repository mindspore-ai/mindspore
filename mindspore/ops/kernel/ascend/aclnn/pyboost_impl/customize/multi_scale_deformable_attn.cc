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
#include <memory>
#include <tuple>
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/cast.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MultiScaleDeformableAttnAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                          const TensorPtr &value_tensor, const TensorPtr &shape_tensor,
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

  OpRunner::InferOpOutput(op, value_tensor_cp, shape_tensor_cp, offset_tensor_cp, locations_tensor_cp,
                          weight_tensor_cp);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), value_tensor_cp, shape_tensor_cp,
                                offset_tensor_cp, locations_tensor_cp, weight_tensor_cp);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, value_tensor_cp, shape_tensor_cp, offset_tensor_cp, locations_tensor_cp, weight_tensor_cp]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, value_tensor_cp, shape_tensor_cp, offset_tensor_cp,
                                   locations_tensor_cp, weight_tensor_cp);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnMultiScaleDeformableAttnFunction, device_context, op->stream_id(), value_tensor_cp,
                   shape_tensor_cp, offset_tensor_cp, locations_tensor_cp, weight_tensor_cp, outputs[0]);
    }));
  op->CreateOutputSimpleInfo();

  if (ori_type == kFloat32) {
    return PyBoostUtils::CastTensor(op->output(0), kNumberTypeFloat32, device::DeviceType::kAscend);
  }
  return PyBoostUtils::CastTensor(op->output(0), kNumberTypeFloat16, device::DeviceType::kAscend);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
