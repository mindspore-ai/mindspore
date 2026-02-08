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

#include "kernel/ascend/aclnn/pyboost_impl/customize/multi_scale_deformable_attn_grad.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/cast.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> MultiScaleDeformableAttnGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &value_tensor, const TensorPtr &shape_tensor,
  const TensorPtr &offset_tensor, const TensorPtr &locations_trans_tensor, const TensorPtr &weight_tensor,
  const TensorPtr &grad_output_tensor) {
  auto ori_type1 = value_tensor->Dtype();
  auto ori_type2 = locations_trans_tensor->Dtype();
  auto ori_type3 = weight_tensor->Dtype();

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

  type = locations_trans_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, locations trans tensor type " << type << " is illegal";
  }

  type = weight_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, weight tensor type " << type << " is illegal";
  }

  type = grad_output_tensor->Dtype();
  if (type != kFloat16 && type != kFloat32) {
    MS_LOG(EXCEPTION) << "For MSDA, grad_output tensor type " << type << " is illegal";
  }

  auto embed_dim = 3;
  if (value_tensor->shape()[embed_dim] % 8 != 0) {
    MS_LOG(EXCEPTION) << "For MSDA, the embed_dim must be a multiple of 8";
  }

  auto value_tensor_cp = PyBoostUtils::CastTensor(value_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);
  auto shape_tensor_cp = PyBoostUtils::CastTensor(shape_tensor, kNumberTypeInt32, device::DeviceType::kAscend);
  auto offset_tensor_cp = PyBoostUtils::CastTensor(offset_tensor, kNumberTypeInt32, device::DeviceType::kAscend);
  auto locations_trans_tensor_cp =
    PyBoostUtils::CastTensor(locations_trans_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);
  auto weight_tensor_cp = PyBoostUtils::CastTensor(weight_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);
  auto grad_output_tensor_cp =
    PyBoostUtils::CastTensor(grad_output_tensor, kNumberTypeFloat32, device::DeviceType::kAscend);

  OpRunner::InferOpOutput(op, value_tensor_cp, shape_tensor_cp, offset_tensor_cp, locations_trans_tensor_cp,
                          weight_tensor_cp, grad_output_tensor_cp);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), value_tensor_cp, shape_tensor_cp,
                                offset_tensor_cp, locations_trans_tensor_cp, weight_tensor_cp, grad_output_tensor_cp);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto shape_shape = shape_tensor_cp->shape();
  auto weight_shape = weight_tensor_cp->shape();
  bool is_equal = (weight_shape[4] == shape_shape[0]);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, value_tensor_cp, shape_tensor_cp, offset_tensor_cp, locations_trans_tensor_cp, weight_tensor_cp,
     grad_output_tensor_cp, is_equal]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, value_tensor_cp, shape_tensor_cp, offset_tensor_cp,
                                   locations_trans_tensor_cp, weight_tensor_cp, grad_output_tensor_cp);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnInplaceZero, device_context, op->stream_id(), outputs[0]);

      if (!is_equal) {
        LAUNCH_ACLNN(aclnnInplaceZero, device_context, op->stream_id(), outputs[1]);
        LAUNCH_ACLNN(aclnnInplaceZero, device_context, op->stream_id(), outputs[2]);
      }

      LAUNCH_ACLNN(aclnnMultiScaleDeformableAttentionGrad, device_context, op->stream_id(), value_tensor_cp,
                   shape_tensor_cp, offset_tensor_cp, locations_trans_tensor_cp, weight_tensor_cp,
                   grad_output_tensor_cp, outputs[0], outputs[1], outputs[2]);
    }));
  op->CreateOutputSimpleInfo();

  auto res_type1 = (ori_type1 == kFloat32) ? kNumberTypeFloat32 : kNumberTypeFloat16;
  auto res_type2 = (ori_type2 == kFloat32) ? kNumberTypeFloat32 : kNumberTypeFloat16;
  auto res_type3 = (ori_type3 == kFloat32) ? kNumberTypeFloat32 : kNumberTypeFloat16;

  auto output_tensor1 = PyBoostUtils::CastTensor(op->output(0), res_type1, device::DeviceType::kAscend);
  auto output_tensor2 = PyBoostUtils::CastTensor(op->output(1), res_type2, device::DeviceType::kAscend);
  auto output_tensor3 = PyBoostUtils::CastTensor(op->output(2), res_type3, device::DeviceType::kAscend);

  return std::make_tuple(output_tensor1, output_tensor2, output_tensor3);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
