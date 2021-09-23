/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/adam_delta_cpu_kernel.h"

#include <vector>
#include <string>
#include <memory>

#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdamDeltaInputsNum = 9;
constexpr size_t kAdamDeltaOutputsNum = 1;
}  // namespace

template <typename T>
void AdamDeltaCPUKernel::LaunchAdamDelta(T *delta, T *m, T *v, float lr, float beta1, float beta2, float epsilon,
                                         const T *gradient, size_t size) {
  std::function<void(size_t, size_t)> task;
  if (dtype_ == kNumberTypeFloat32) {
    task = [this, delta, m, v, lr, beta1, beta2, epsilon, gradient](size_t start, size_t end) {
      (void)AdamDeltaFp32(delta, m, v, lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
    };
  } else {
    task = [this, delta, m, v, lr, beta1, beta2, epsilon, gradient](size_t start, size_t end) {
      for (size_t c1 = start; c1 < end; ++c1) {
        m[c1] *= beta1;
        m[c1] += (1 - beta1) * gradient[c1];
        v[c1] *= beta2;
        v[c1] += (1 - beta2) * gradient[c1] * gradient[c1];
        if (use_nesterov_) {
          delta[c1] = -lr * (m[c1] * beta1 + (1 - beta1) * gradient[c1]) / (std::sqrt(v[c1]) + epsilon);
        } else {
          delta[c1] = -lr * m[c1] / (std::sqrt(v[c1]) + epsilon);
        }
      }
    };
  }
  CPUKernelUtils::ParallelFor(task, size);
}

void AdamDeltaCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> delta_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<size_t> m_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> v_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> grad_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 8);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (!IsSameShape(delta_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "Delta and m should have the same shape";
  }
  if (!IsSameShape(delta_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "Delta and v should have the same shape";
  }
  if (!IsSameShape(delta_shape, grad_shape)) {
    MS_LOG(EXCEPTION) << "Delta and grad should have the same shape";
  }
  if (delta_shape.empty()) {
    MS_LOG(EXCEPTION) << "Delta must be at least 1D";
  }
  elem_num_ = 1;
  for (size_t i = 0; i < delta_shape.size(); ++i) {
    elem_num_ *= delta_shape[i];
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "Invalid delta shape";
  }
  if (AnfAlgo::HasNodeAttr(USE_NESTEROV, kernel_node)) {
    use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
  }
}

void AdamDeltaCPUKernel::CheckParams(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamDeltaInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamDeltaOutputsNum, kernel_name_);

  size_t elem_size = elem_num_ * 4;
  std::vector<size_t> expect_sizes = {elem_size, elem_size, 4, 4, 4, 4, 4, 4, elem_size};
  std::vector<std::string> input_names = {"m",     "v",     "beta1_power", "beta2_power", "lr",
                                          "beta1", "beta2", "epsilon",     "grad"};
  for (size_t i = 0; i < kAdamDeltaInputsNum; ++i) {
    if (inputs[i]->size != expect_sizes[i]) {
      MS_LOG(EXCEPTION) << "Error input " << input_names[i] << " size!";
    }
  }
  if (outputs.size() < 1 || outputs[0]->size != elem_size) {
    MS_LOG(EXCEPTION) << "Error output delta size!";
  }
}

bool AdamDeltaCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CheckParams(inputs, outputs);
  auto m = reinterpret_cast<float *>(inputs[0]->addr);
  auto v = reinterpret_cast<float *>(inputs[1]->addr);
  auto beta1_power = reinterpret_cast<float *>(inputs[2]->addr)[0];
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "The beta1_power should not be 1";
  }
  auto beta2_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  auto lr = reinterpret_cast<float *>(inputs[4]->addr)[0];
  auto beta1 = reinterpret_cast<float *>(inputs[5]->addr)[0];
  auto beta2 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  auto epsilon = reinterpret_cast<float *>(inputs[7]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[8]->addr);
  auto delta = reinterpret_cast<float *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(m);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(grad);
  MS_EXCEPTION_IF_NULL(delta);

  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
  LaunchAdamDelta<float>(delta, m, v, lr, beta1, beta2, epsilon, grad, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
