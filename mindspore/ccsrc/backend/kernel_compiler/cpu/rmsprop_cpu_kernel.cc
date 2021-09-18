/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/errorcode.h"
#include "nnacl/fp32/rmsprop_fp32.h"
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/rmsprop_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCenteredRMSPropInputsNum = 9;
constexpr size_t kRMSPropInputsNum = 5;
}  // namespace

template <typename T>
void RMSPropCPUKernel<T>::LaunchRMSPropUnuseCenter(T *variable, T *mean_square, T *moment, T *gradients,
                                                   float *learning_rate) {
  std::function<void(size_t, size_t)> task;
  if (dtype_ == kNumberTypeFloat32) {
    task = [this, &variable, &mean_square, &moment, &gradients, &learning_rate](size_t start, size_t end) {
      (void)RMSPropUnuseCenterFp32(variable, mean_square, moment, gradients, momentum_, learning_rate[0], decay_,
                                   epsilon_, start, end);
    };
  } else {
    task = [this, &variable, &mean_square, &moment, &gradients, &learning_rate](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        mean_square[i] += (gradients[i] * gradients[i] - mean_square[i]) * (1.0 - decay_);
        moment[i] = moment[i] * momentum_ + (gradients[i] * learning_rate[0]) / sqrt(mean_square[i] + epsilon_);
        variable[i] -= moment[i];
      }
    };
  }
  CPUKernelUtils::ParallelFor(task, size_);
}

template <typename T>
void RMSPropCPUKernel<T>::LaunchRMSPropUseCenter(T *variable, T *mean_square, T *moment, T *gradients,
                                                 T *mean_gradients, float *momentum, float *learning_rate, float *decay,
                                                 float *epsilon) {
  std::function<void(size_t, size_t)> task;
  if (dtype_ == kNumberTypeFloat32) {
    task = [&](size_t start, size_t end) {
      (void)RMSPropUseCenterFp32(variable, mean_square, moment, gradients, mean_gradients, momentum[0],
                                 learning_rate[0], decay[0], epsilon[0], start, end);
    };
  } else {
    task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        mean_square[i] += (gradients[i] * gradients[i] - mean_square[i]) * (1.0 - decay[0]);
        mean_gradients[i] += (gradients[i] - mean_gradients[i]) * (1.0 - decay[0]);
        auto denom = (mean_square[i] - mean_gradients[i] * mean_gradients[i]) + epsilon[0];
        if (denom > 0) {
          moment[i] = moment[i] * momentum[0] + (gradients[i] * learning_rate[0]) / sqrt(denom);
          variable[i] -= moment[i];
        }
      }
    };
  }
  CPUKernelUtils::ParallelFor(task, size_);
}

template <typename T>
void RMSPropCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto node_name = AnfAlgo::GetCNodeName(kernel_node);
  if (node_name == "ApplyCenteredRMSProp") {
    use_center_ = true;
  }

  if (node_name == "ApplyRMSProp") {
    decay_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "rho");
    momentum_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "momentum");
    epsilon_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  }
  auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  for (auto &dim : input_shape) {
    size_ *= dim;
  }
}

template <typename T>
bool RMSPropCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &) {
  if (!use_center_) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRMSPropInputsNum, kernel_name_);
    float *variable = reinterpret_cast<float *>(inputs[0]->addr);
    float *mean_square = reinterpret_cast<float *>(inputs[1]->addr);
    float *moment = reinterpret_cast<float *>(inputs[2]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[3]->addr);
    float *gradients = reinterpret_cast<float *>(inputs[4]->addr);

    size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
    MS_LOG(INFO) << "RMSPropCPUKernel lens:" << lens << " size_:" << size_;
    LaunchRMSPropUnuseCenter(variable, mean_square, moment, gradients, learning_rate);
  } else {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCenteredRMSPropInputsNum, kernel_name_);
    T *variable = reinterpret_cast<float *>(inputs[0]->addr);
    T *mean_gradients = reinterpret_cast<float *>(inputs[1]->addr);
    T *mean_square = reinterpret_cast<float *>(inputs[2]->addr);
    T *moment = reinterpret_cast<float *>(inputs[3]->addr);
    T *gradients = reinterpret_cast<float *>(inputs[4]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[5]->addr);
    float *decay = reinterpret_cast<float *>(inputs[6]->addr);
    float *momentum = reinterpret_cast<float *>(inputs[7]->addr);
    float *epsilon = reinterpret_cast<float *>(inputs[8]->addr);

    size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
    MS_LOG(INFO) << "RMSPropCPUKernel lens:" << lens << " size_:" << size_;
    LaunchRMSPropUseCenter(variable, mean_square, moment, gradients, mean_gradients, momentum, learning_rate, decay,
                           epsilon);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
