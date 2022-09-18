/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/batch_norm_grad_grad_cpu_kernel.h"
#include <cmath>
#include <numeric>
#include <vector>
#include <functional>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/batch_norm_grad_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 8;
constexpr size_t kOutputsNum = 3;
constexpr float num_1 = 1.0;
constexpr float num_3 = 3.0;
}  // namespace

bool BatchNormGradGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto op = std::dynamic_pointer_cast<ops::BatchNormGradGrad>(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
  }
  is_training_ = op->get_is_training();
  epsilon_ = op->get_epsilon();
  data_format_ = op->get_format();
  if (epsilon_ <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'epsilon' must be greater than 0.";
  }
  if (data_format_ != kOpFormat_NHWC && data_format_ != kOpFormat_NCHW) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'data_format' must be NHWC or NCHW, but got "
                             << data_format_;
  }
  data_format_ = (data_format_ == kOpFormat_NHWC) ? "NHWC" : "NCHW";

  std::vector<int64_t> x_shape = inputs.at(kIndex0)->GetShapeVector();
  std::transform(x_shape.begin(), x_shape.end(), std::back_inserter(x_shape_), LongToSize);
  std::vector<int64_t> scale_shape = inputs.at(kIndex2)->GetShapeVector();
  std::transform(scale_shape.begin(), scale_shape.end(), std::back_inserter(scale_shape_), LongToSize);

  x_num_ = static_cast<int>(std::accumulate(x_shape_.begin(), x_shape_.end(), 1, std::multiplies<size_t>()));
  N_num_ = static_cast<int>(x_shape_[0]);
  C_num_ = static_cast<int>(std::accumulate(scale_shape_.begin(), scale_shape_.end(), 1, std::multiplies<size_t>()));
  CHW_num_ = static_cast<int>(x_num_ / N_num_);
  NHW_num_ = static_cast<int>(x_num_ / C_num_);
  HW_num_ = static_cast<int>(NHW_num_ / N_num_);
  M_ = static_cast<float>(NHW_num_);

  kernel_func_ = func_list_[index].second;
  return true;
}

int BatchNormGradGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> dy_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<int64_t> reserve_space_1_shape = inputs.at(kIndex3)->GetShapeVector();
  std::vector<int64_t> reserve_space_2_shape = inputs.at(kIndex4)->GetShapeVector();
  std::vector<int64_t> ddx_shape = inputs.at(kIndex5)->GetShapeVector();
  std::vector<int64_t> ddscale_shape = inputs.at(kIndex6)->GetShapeVector();
  std::vector<int64_t> ddoffset_shape = inputs.at(kIndex7)->GetShapeVector();
  if (x_shape_.size() != kDim4) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', dim of x must be equal 4, but got" << x_shape_.size();
  }
  if (x_shape_ != dy_shape || x_shape_ != ddx_shape) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'x' 'dy' 'ddx' must have the same shape.";
  }
  if (scale_shape_ != reserve_space_1_shape || scale_shape_ != reserve_space_2_shape || scale_shape_ != ddscale_shape ||
      scale_shape_ != ddoffset_shape) {
    MS_EXCEPTION(ValueError)
      << "For '" << kernel_name_
      << "', 'scale' 'reserve_space_1' 'reserve_space_2' 'ddscale' 'ddoffset' must have the same shape";
  }
  if ((data_format_ == "NHWC" && x_shape_[kIndex3] != C_num_) ||
      (data_format_ == "NCHW" && x_shape_[kIndex1] != C_num_)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the size of 1D tensor should be equal to the size of C dim of 'x'";
  }
  size_t float_type_size = sizeof(float);
  workspace_size_list_.clear();
  if (is_training_) {
    // Training model: total workspace number = 12; 6 x_num_ * float_type_size, 6 C_num_ * float_type_size.
    // kIndex0-kIndex5: This workspace size used for new float[x_num_].
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    // kIndex6-kIndex11: This workspace size used for new float[C_num_].
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
  } else {
    // Inference model: total workspace number = 5; 4 x_num_ * float_type_size, 1 C_num_ * float_type_size.
    // kIndex0-kIndex3: This workspace size used for new float[x_num_].
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    (void)workspace_size_list_.emplace_back(x_num_ * float_type_size);
    // kIndex4: This workspace size used for new float[C_num_].
    (void)workspace_size_list_.emplace_back(C_num_ * float_type_size);
  }
  return ret;
}

template <typename T>
bool BatchNormGradGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &workspace,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // fp32
  for (int j = 0; j < C_num_; j++) {
    if (*(reserve_space_2 + j) < 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'reserve_space_2' must be no less than zero.";
    }
  }
  if (is_training_ && data_format_ == "NHWC") {
    TrainingComputeNHWC<T>(inputs, workspace, outputs);
  } else if (!is_training_ && data_format_ == "NHWC") {
    InferenceComputeNHWC<T>(inputs, workspace, outputs);
  } else if (is_training_ && data_format_ == "NCHW") {
    TrainingComputeNCHW<T>(inputs, workspace, outputs);
  } else if (!is_training_ && data_format_ == "NCHW") {
    InferenceComputeNCHW<T>(inputs, workspace, outputs);
  }

  return true;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingComputeNHWC(const std::vector<kernel::AddressPtr> &inputs,
                                                        const std::vector<kernel::AddressPtr> &workspace,
                                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ori = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto reserve_space_1 = reinterpret_cast<float *>(inputs.at(kIndex3)->addr);  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);

  // change dtype from 'T' to 'fp32'
  float *x = reinterpret_cast<float *>(workspace.at(kIndex0)->addr);
  float *dy = reinterpret_cast<float *>(workspace.at(kIndex1)->addr);
  float *ddx = reinterpret_cast<float *>(workspace.at(kIndex2)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  // create intermediate variables
  float *x_hat = reinterpret_cast<float *>(workspace.at(kIndex3)->addr);
  float *inv_std = reinterpret_cast<float *>(workspace.at(kIndex11)->addr);

  for (int j = 0; j < C_num_; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon_);
  }

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }
  TrainingNHWCCalculateDx<T>(inputs, workspace, outputs, x_hat, inv_std);
  TrainingNHWCCalculateDdy<T>(inputs, workspace, outputs, x_hat, inv_std);
  TrainingNHWCCalculateDscale<T>(inputs, workspace, outputs, x_hat, inv_std);
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::InferenceComputeNHWC(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &workspace,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ori = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(inputs.at(kIndex3)->addr);  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);   // fp32
  auto ddoffset = reinterpret_cast<float *>(inputs.at(kIndex7)->addr);  // fp32
  auto dx = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto ddy = reinterpret_cast<T *>(outputs.at(kIndex1)->addr);
  auto dscale = reinterpret_cast<float *>(outputs.at(kIndex2)->addr);  // fp32

  // change dtype from 'T' to 'fp32'
  float *x = reinterpret_cast<float *>(workspace.at(kIndex0)->addr);
  float *dy = reinterpret_cast<float *>(workspace.at(kIndex1)->addr);
  float *ddx = reinterpret_cast<float *>(workspace.at(kIndex2)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  // create intermediate variables
  float *x_hat = reinterpret_cast<float *>(workspace.at(kIndex3)->addr);
  float *inv_std = reinterpret_cast<float *>(workspace.at(kIndex4)->addr);

  for (int j = 0; j < C_num_; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon_);
  }

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // initialize dscale
  for (int j = 0; j < C_num_; j++) {
    *(dscale + j) = 0;
  }

  // calculate dx, ddy, dscale
  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(dx + index) = static_cast<T>((*(ddscale + j)) * (*(inv_std + j)) * (*(dy + index)));
        *(ddy + index) = static_cast<T>((*(ddx + index)) * (*(inv_std + j)) * (*(scale + j)) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + *(ddoffset + j));
        *(dscale + j) += (*(ddx + index)) * (*(dy + index)) * (*(inv_std + j));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingComputeNCHW(const std::vector<kernel::AddressPtr> &inputs,
                                                        const std::vector<kernel::AddressPtr> &workspace,
                                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ori = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto reserve_space_1 = reinterpret_cast<float *>(inputs.at(kIndex3)->addr);  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);

  // change dtype from 'T' to 'fp32'
  float *x = reinterpret_cast<float *>(workspace.at(kIndex0)->addr);
  float *dy = reinterpret_cast<float *>(workspace.at(kIndex1)->addr);
  float *ddx = reinterpret_cast<float *>(workspace.at(kIndex2)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  // create intermediate variables
  float *x_hat = reinterpret_cast<float *>(workspace.at(kIndex3)->addr);
  float *inv_std = reinterpret_cast<float *>(workspace.at(kIndex11)->addr);

  for (int j = 0; j < C_num_; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon_);
  }

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }
  TrainingNCHWCalculateDx<T>(inputs, workspace, outputs, x_hat, inv_std);
  TrainingNCHWCalculateDdy<T>(inputs, workspace, outputs, x_hat, inv_std);
  TrainingNCHWCalculateDscale<T>(inputs, workspace, outputs, x_hat, inv_std);
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::InferenceComputeNCHW(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &workspace,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ori = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(inputs.at(kIndex3)->addr);  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);   // fp32
  auto ddoffset = reinterpret_cast<float *>(inputs.at(kIndex7)->addr);  // fp32
  auto dx = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto ddy = reinterpret_cast<T *>(outputs.at(kIndex1)->addr);
  auto dscale = reinterpret_cast<float *>(outputs.at(kIndex2)->addr);  // fp32

  // change dtype from 'T' to 'fp32'
  float *x = reinterpret_cast<float *>(workspace.at(kIndex0)->addr);
  float *dy = reinterpret_cast<float *>(workspace.at(kIndex1)->addr);
  float *ddx = reinterpret_cast<float *>(workspace.at(kIndex2)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  // create intermediate variables
  float *x_hat = reinterpret_cast<float *>(workspace.at(kIndex3)->addr);
  float *inv_std = reinterpret_cast<float *>(workspace.at(kIndex4)->addr);

  for (int j = 0; j < C_num_; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon_);
  }

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // initialize dscale
  for (int j = 0; j < C_num_; j++) {
    *(dscale + j) = 0;
  }

  // calculate dx, ddy, dscale
  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(dx + index) = static_cast<T>((*(ddscale + j)) * (*(inv_std + j)) * (*(dy + index)));
        *(ddy + index) = static_cast<T>((*(ddx + index)) * (*(inv_std + j)) * (*(scale + j)) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + *(ddoffset + j));
        *(dscale + j) += (*(ddx + index)) * (*(dy + index)) * (*(inv_std + j));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNHWCCalculateDx(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<kernel::AddressPtr> &workspace,
                                                            const std::vector<kernel::AddressPtr> &outputs,
                                                            float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);            // fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);  // fp32
  auto dx = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  float *dx_term = reinterpret_cast<float *>(workspace.at(kIndex4)->addr);
  float *scale_term = reinterpret_cast<float *>(workspace.at(kIndex5)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(dx_term + index) =
          (*(scale + j)) / ((*(reserve_space_2 + j)) + epsilon_) *
          (((*(x_hat + index)) *
            ((*(sum_ddx + j)) * (*(sum_dy + j)) / M_ - (*(sum_dy_ddx + j)) +
             num_3 * (*(sum_dy_x_hat + j)) * (*(sum_ddx_x_hat + j)) / M_) /
            M_) +
           ((*(sum_ddx_x_hat + j)) * ((*(sum_dy + j)) / M_ - (static_cast<float>(*(dy_ori + index)))) / M_) +
           (*(sum_dy_x_hat + j)) * ((*(sum_ddx + j)) / M_ - (static_cast<float>(*(ddx_ori + index)))) / M_);
        *(scale_term + index) = (*(ddscale + j)) * (*(inv_std + j)) *
                                ((static_cast<float>(*(dy_ori + index))) - (*(sum_dy + j)) / M_ -
                                 (*(sum_dy_x_hat + j)) / M_ * (*(x_hat + index)));
      }
    }
  }

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(dx + index) = static_cast<T>((*(dx_term + index)) + (*(scale_term + index)));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNHWCCalculateDdy(const std::vector<kernel::AddressPtr> &inputs,
                                                             const std::vector<kernel::AddressPtr> &workspace,
                                                             const std::vector<kernel::AddressPtr> &outputs,
                                                             float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);  // fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);   // fp32
  auto ddoffset = reinterpret_cast<float *>(inputs.at(kIndex7)->addr);  // fp32
  auto ddy = reinterpret_cast<T *>(outputs.at(kIndex1)->addr);

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(ddy + index) = static_cast<T>((*(scale + j)) * (*(inv_std + j)) / M_ *
                                          (M_ * (static_cast<float>(*(ddx_ori + index))) - (*(sum_ddx + j)) -
                                           (*(x_hat + index)) * (*(sum_ddx_x_hat + j))) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + (*(ddoffset + j)));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNHWCCalculateDscale(const std::vector<kernel::AddressPtr> &inputs,
                                                                const std::vector<kernel::AddressPtr> &workspace,
                                                                const std::vector<kernel::AddressPtr> &outputs,
                                                                float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto dscale = reinterpret_cast<float *>(outputs.at(kIndex2)->addr);  // fp32

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  for (int j = 0; j < C_num_; j++) {
    *(dscale + j) = 0;
  }

  for (int i = 0; i < N_num_; i++) {
    for (int k = 0; k < HW_num_; k++) {
      for (int j = 0; j < C_num_; j++) {
        int index = i * HW_num_ * C_num_ + k * C_num_ + j;
        *(dscale + j) += (static_cast<float>(*(ddx_ori + index))) * (*(inv_std + j)) *
                         ((static_cast<float>(*(dy_ori + index))) - (*(sum_dy + j)) / M_ -
                          (*(sum_dy_x_hat + j)) / M_ * (*(x_hat + index)));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNCHWCalculateDx(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<kernel::AddressPtr> &workspace,
                                                            const std::vector<kernel::AddressPtr> &outputs,
                                                            float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);            // fp32
  auto reserve_space_2 = reinterpret_cast<float *>(inputs.at(kIndex4)->addr);  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);  // fp32
  auto dx = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  float *dx_term = reinterpret_cast<float *>(workspace.at(kIndex4)->addr);
  float *scale_term = reinterpret_cast<float *>(workspace.at(kIndex5)->addr);

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(dx_term + index) =
          (*(scale + j)) / ((*(reserve_space_2 + j)) + epsilon_) *
          (((*(x_hat + index)) *
            ((*(sum_ddx + j)) * (*(sum_dy + j)) / M_ - (*(sum_dy_ddx + j)) +
             num_3 * (*(sum_dy_x_hat + j)) * (*(sum_ddx_x_hat + j)) / M_) /
            M_) +
           ((*(sum_ddx_x_hat + j)) * ((*(sum_dy + j)) / M_ - (static_cast<float>(*(dy_ori + index)))) / M_) +
           (*(sum_dy_x_hat + j)) * ((*(sum_ddx + j)) / M_ - (static_cast<float>(*(ddx_ori + index)))) / M_);
        *(scale_term + index) = (*(ddscale + j)) * (*(inv_std + j)) *
                                ((static_cast<float>(*(dy_ori + index))) - (*(sum_dy + j)) / M_ -
                                 (*(sum_dy_x_hat + j)) / M_ * (*(x_hat + index)));
      }
    }
  }

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(dx + index) = static_cast<T>((*(dx_term + index)) + (*(scale_term + index)));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNCHWCalculateDdy(const std::vector<kernel::AddressPtr> &inputs,
                                                             const std::vector<kernel::AddressPtr> &workspace,
                                                             const std::vector<kernel::AddressPtr> &outputs,
                                                             float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto scale = reinterpret_cast<float *>(inputs.at(kIndex2)->addr);  // fp32
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto ddscale = reinterpret_cast<float *>(inputs.at(kIndex6)->addr);   // fp32
  auto ddoffset = reinterpret_cast<float *>(inputs.at(kIndex7)->addr);  // fp32
  auto ddy = reinterpret_cast<T *>(outputs.at(kIndex1)->addr);

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(ddy + index) = static_cast<T>((*(scale + j)) * (*(inv_std + j)) / M_ *
                                          (M_ * (static_cast<float>(*(ddx_ori + index))) - (*(sum_ddx + j)) -
                                           (*(x_hat + index)) * (*(sum_ddx_x_hat + j))) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + (*(ddoffset + j)));
      }
    }
  }
  return;
}

template <typename T>
void BatchNormGradGradCpuKernelMod::TrainingNCHWCalculateDscale(const std::vector<kernel::AddressPtr> &inputs,
                                                                const std::vector<kernel::AddressPtr> &workspace,
                                                                const std::vector<kernel::AddressPtr> &outputs,
                                                                float *x_hat, float *inv_std) {
  auto dy_ori = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto ddx_ori = reinterpret_cast<T *>(inputs.at(kIndex5)->addr);
  auto dscale = reinterpret_cast<float *>(outputs.at(kIndex2)->addr);  // fp32

  // create intermediate variables
  float *sum_dy = reinterpret_cast<float *>(workspace.at(kIndex6)->addr);
  float *sum_dy_x_hat = reinterpret_cast<float *>(workspace.at(kIndex7)->addr);
  float *sum_ddx = reinterpret_cast<float *>(workspace.at(kIndex8)->addr);
  float *sum_ddx_x_hat = reinterpret_cast<float *>(workspace.at(kIndex9)->addr);
  float *sum_dy_ddx = reinterpret_cast<float *>(workspace.at(kIndex10)->addr);

  // initialize
  for (int j = 0; j < C_num_; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(sum_dy + j) += static_cast<float>(*(dy_ori + index));
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(dy_ori + index)));
        *(sum_ddx + j) += static_cast<float>(*(ddx_ori + index));
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (static_cast<float>(*(ddx_ori + index)));
        *(sum_dy_ddx + j) += (static_cast<float>(*(dy_ori + index))) * (static_cast<float>(*(ddx_ori + index)));
      }
    }
  }

  for (int j = 0; j < C_num_; j++) {
    *(dscale + j) = 0;
  }

  for (int i = 0; i < N_num_; i++) {
    for (int j = 0; j < C_num_; j++) {
      for (int k = 0; k < HW_num_; k++) {
        int index = i * CHW_num_ + j * HW_num_ + k;
        *(dscale + j) += (static_cast<float>(*(ddx_ori + index))) * (*(inv_std + j)) *
                         ((static_cast<float>(*(dy_ori + index))) - (*(sum_dy + j)) / M_ -
                          (*(sum_dy_x_hat + j)) / M_ * (*(x_hat + index)));
      }
    }
  }
  return;
}
std::vector<std::pair<KernelAttr, BatchNormGradGradCpuKernelMod::BatchNormGradGradFunc>>
  BatchNormGradGradCpuKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32)    // x
                                                  .AddInputAttr(kNumberTypeFloat32)    // dy
                                                  .AddInputAttr(kNumberTypeFloat32)    // scale
                                                  .AddInputAttr(kNumberTypeFloat32)    // reserve_space_1
                                                  .AddInputAttr(kNumberTypeFloat32)    // reserve_space_2
                                                  .AddInputAttr(kNumberTypeFloat32)    // ddx
                                                  .AddInputAttr(kNumberTypeFloat32)    // ddscale
                                                  .AddInputAttr(kNumberTypeFloat32)    // ddoffset
                                                  .AddOutputAttr(kNumberTypeFloat32)   // dx
                                                  .AddOutputAttr(kNumberTypeFloat32)   // ddy
                                                  .AddOutputAttr(kNumberTypeFloat32),  // dscale
                                                &BatchNormGradGradCpuKernelMod::LaunchKernel<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat16)    // x
                                                  .AddInputAttr(kNumberTypeFloat16)    // dy
                                                  .AddInputAttr(kNumberTypeFloat32)    // scale
                                                  .AddInputAttr(kNumberTypeFloat32)    // reserve_space_1
                                                  .AddInputAttr(kNumberTypeFloat32)    // reserve_space_2
                                                  .AddInputAttr(kNumberTypeFloat16)    // ddx
                                                  .AddInputAttr(kNumberTypeFloat32)    // ddscale
                                                  .AddInputAttr(kNumberTypeFloat32)    // ddoffset
                                                  .AddOutputAttr(kNumberTypeFloat16)   // dx
                                                  .AddOutputAttr(kNumberTypeFloat16)   // ddy
                                                  .AddOutputAttr(kNumberTypeFloat32),  // dscale
                                                &BatchNormGradGradCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> BatchNormGradGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BatchNormGradGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNormGradGrad, BatchNormGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
