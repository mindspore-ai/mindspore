/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/rmsprop_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include <functional>
#include "nnacl/fp32/rmsprop_fp32.h"
#include "utils/ms_utils.h"
#include "ops/apply_rms_prop.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCenteredRMSPropInputsNum = 9;
constexpr size_t kRMSPropInputsNum = 5;
constexpr auto kApplyRMSProp = "ApplyRMSProp";
constexpr auto kApplyCenteredRMSProp = "ApplyCenteredRMSProp";
constexpr auto kNumberZero = 0;
constexpr auto kNumberOne = 1;
constexpr auto kNumberTwo = 2;
constexpr auto kNumberThree = 3;
constexpr auto kNumberFour = 4;
constexpr auto kNumberFive = 5;
constexpr auto kNumberSix = 6;
constexpr auto kNumberSeven = 7;
constexpr auto kNumberEight = 8;
}  // namespace

template <typename T>
void RMSPropCpuKernelMod::LaunchRMSPropUnuseCenter(T *variable, T *mean_square, T *moment, T *gradients,
                                                   float *learning_rate) {
  std::function<void(size_t, size_t)> task;
  for (int64_t b = 0; b < batch_size_; b++) {
    if (dtype_ == kNumberTypeFloat32) {
      task = [this, &variable, &mean_square, &moment, &gradients, &learning_rate](size_t start, size_t end) {
        (void)RMSPropUnuseCenterFp32(variable, mean_square, moment, gradients, momentum_, learning_rate[0], decay_,
                                     epsilon_, start, end);
      };
    } else {
      // multithreading
      task = [this, &variable, &mean_square, &moment, &gradients, &learning_rate](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          mean_square[i] += (gradients[i] * gradients[i] - mean_square[i]) * (1.0 - decay_);
          moment[i] = moment[i] * momentum_ + (gradients[i] * learning_rate[0]) / sqrt(mean_square[i] + epsilon_);
          variable[i] -= moment[i];
        }
      };
    }
    ParallelLaunchAutoSearch(task, LongToSize(input_elements_), this, &parallel_search_info_);
    variable = variable + input_elements_;
    mean_square = mean_square + input_elements_;
    moment = moment + input_elements_;
    gradients = gradients + input_elements_;
    learning_rate++;
  }
}

template <typename T>
void RMSPropCpuKernelMod::LaunchRMSPropUseCenter(T *variable, T *mean_square, T *moment, T *gradients,
                                                 T *mean_gradients, float *momentum, float *learning_rate, float *decay,
                                                 float *epsilon) {
  std::function<void(size_t, size_t)> task;
  for (int64_t b = 0; b < batch_size_; b++) {
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
    ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_);
    variable = variable + input_elements_;
    mean_square = mean_square + input_elements_;
    moment = moment + input_elements_;
    gradients = gradients + input_elements_;
    mean_gradients = mean_gradients + input_elements_;
    momentum++;
    learning_rate++;
    decay++;
    epsilon++;
  }
}

bool RMSPropCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();

  auto node_name = base_operator->name();
  if (node_name == "ApplyCenteredRMSProp") {
    use_center_ = true;
  }
  if (node_name == "ApplyRMSProp") {
    auto kernel_ptr = std::make_shared<ops::ApplyRMSProp>(base_operator->GetPrim());
    decay_ = kernel_ptr->get_attr("rho");
    momentum_ = kernel_ptr->get_attr("momentum");
    epsilon_ = kernel_ptr->get_attr("epsilon");
  }

  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "RMSProp does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[kernel_name_][index].second;
  return true;
}

int RMSPropCpuKernelMod::CalElements(std::vector<int64_t> var_shape, std::vector<int64_t> lr_shape, int ret) {
  if (batch_rank_ == 0 && lr_shape.size() != 0 && lr_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be 0 or 1, but got the shape of 'lr': " << Vector2Str(lr_shape)
                  << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }
  if (batch_rank_ < 0 || (batch_rank_ > 0 && lr_shape.size() != LongToSize(batch_rank_))) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), int64_t(1), std::multiplies<int64_t>());
  }
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;
  return ret;
}

int RMSPropCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), use_center_ ? kCenteredRMSPropInputsNum : kRMSPropInputsNum, kernel_name_);
  auto var_shape = inputs[kNumberZero]->GetShapeVector();
  auto grad_shape = inputs[kNumberFour]->GetShapeVector();
  auto mean_square_shape = inputs[use_center_ ? kNumberTwo : kNumberOne]->GetShapeVector();
  auto moment_shape = inputs[use_center_ ? kNumberThree : kNumberTwo]->GetShapeVector();
  auto lr_shape = inputs[use_center_ ? kNumberFive : kNumberThree]->GetShapeVector();
  ShapeArray shapes{var_shape, mean_square_shape, moment_shape, grad_shape};

  if (use_center_) {
    auto mean_gradients_shape = inputs[kNumberOne]->GetShapeVector();
    shapes.push_back(mean_gradients_shape);
    if (!CheckShapesSame(shapes)) {
      MS_LOG(EXCEPTION)
        << "For " << kernel_name_
        << ", the var shape, mean_gradients shape, mean_square shape, moment shape, and grad shape should "
        << "be same, but got var shape: " << Vector2Str(var_shape)
        << " mean_gradients shape: " << Vector2Str(mean_gradients_shape)
        << " mean_square shape: " << Vector2Str(mean_square_shape) << ", moment shape: " << Vector2Str(moment_shape)
        << ", grad shape: " << Vector2Str(grad_shape);
    }
  } else {
    if (!CheckShapesSame(shapes)) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_
                        << ", the var shape, mean_square shape, moment shape, and grad shape should "
                        << "be same, but got var shape: " << Vector2Str(var_shape)
                        << " mean_square shape: " << Vector2Str(mean_square_shape)
                        << ", moment shape: " << Vector2Str(moment_shape) << ", grad shape: " << Vector2Str(grad_shape);
    }
  }
  return CalElements(var_shape, lr_shape, ret);
}

template <typename T>
bool RMSPropCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &) {
  size_ = inputs[0]->size / sizeof(T);
  if (!use_center_) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRMSPropInputsNum, kernel_name_);
    float *variable = reinterpret_cast<float *>(inputs[kNumberZero]->addr);
    float *mean_square = reinterpret_cast<float *>(inputs[kNumberOne]->addr);
    float *moment = reinterpret_cast<float *>(inputs[kNumberTwo]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[kNumberThree]->addr);
    float *gradients = reinterpret_cast<float *>(inputs[kNumberFour]->addr);

    size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
    MS_LOG(INFO) << "RMSPropCpuKernelMod lens:" << lens << " size_:" << size_;
    LaunchRMSPropUnuseCenter<T>(variable, mean_square, moment, gradients, learning_rate);
  } else {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCenteredRMSPropInputsNum, kernel_name_);
    T *variable = reinterpret_cast<float *>(inputs[kNumberZero]->addr);
    T *mean_gradients = reinterpret_cast<float *>(inputs[kNumberOne]->addr);
    T *mean_square = reinterpret_cast<float *>(inputs[kNumberTwo]->addr);
    T *moment = reinterpret_cast<float *>(inputs[kNumberThree]->addr);
    T *gradients = reinterpret_cast<float *>(inputs[kNumberFour]->addr);
    float *learning_rate = reinterpret_cast<float *>(inputs[kNumberFive]->addr);
    float *decay = reinterpret_cast<float *>(inputs[kNumberSix]->addr);
    float *momentum = reinterpret_cast<float *>(inputs[kNumberSeven]->addr);
    float *epsilon = reinterpret_cast<float *>(inputs[kNumberEight]->addr);

    size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
    MS_LOG(INFO) << "RMSPropCpuKernelMod lens:" << lens << " size_:" << size_;
    LaunchRMSPropUseCenter<T>(variable, mean_square, moment, gradients, mean_gradients, momentum, learning_rate, decay,
                              epsilon);
  }
  return true;
}

std::map<std::string, std::vector<std::pair<KernelAttr, RMSPropCpuKernelMod::RMSPropFunc>>>
  RMSPropCpuKernelMod::func_list_ = {{kApplyRMSProp,
                                      {{KernelAttr()
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddOutputAttr(kNumberTypeFloat32),
                                        &RMSPropCpuKernelMod::LaunchKernel<float>}}},
                                     {kApplyCenteredRMSProp,
                                      {{KernelAttr()
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddInputAttr(kNumberTypeFloat32)
                                          .AddOutputAttr(kNumberTypeFloat32),
                                        &RMSPropCpuKernelMod::LaunchKernel<float>}}}};

std::vector<KernelAttr> RMSPropCpuKernelMod::GetOpSupport() {
  auto iter = func_list_.find(kernel_type_);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "RMSProp cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RMSPropFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ApplyRMSProp,
                                 []() { return std::make_shared<RMSPropCpuKernelMod>(kApplyRMSProp); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ApplyCenteredRMSProp,
                                 []() { return std::make_shared<RMSPropCpuKernelMod>(kApplyCenteredRMSProp); });
}  // namespace kernel
}  // namespace mindspore
