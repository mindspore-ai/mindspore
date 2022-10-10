/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/fused_ada_factor_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include "mindspore/core/ops/fused_ada_factor.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kSizeFloat16 = sizeof(float16);
constexpr size_t kScalarIndex = 0;
constexpr size_t kStandardInputNum = 12;
constexpr size_t kWorkSpaceNum = 3;
constexpr size_t kBatchSize = 1000;
constexpr size_t kLastRowIndex = 1;
constexpr size_t kLastColIndex = 2;
constexpr float kEps = 1e-30;
constexpr size_t kEpsIndex = 0;
constexpr size_t kClipThresholdIndex = 1;
constexpr size_t kBeta1Index = 2;
constexpr size_t kBeta2tIndex = 3;
constexpr size_t kWeightDecayIndex = 4;
constexpr size_t kLearningRateIndex = 5;
constexpr size_t kGradIndex = 6;
constexpr size_t kParamIndex = 7;
constexpr size_t kExpAvgIndex = 8;
constexpr size_t kExpAvgSQRowIndex = 9;
constexpr size_t kExpAvgSQColIndex = 10;
constexpr size_t kExpAvgSQIndex = 11;
constexpr size_t kGlobalNormIndex = 12;
constexpr size_t kWorkSpaceUpdateIndex = 0;
constexpr size_t kWorkSpaceRFactorIndex = 1;
constexpr size_t kWorkSpaceCFactorIndex = 2;
}  // namespace

bool FusedAdaFactorCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  param_dtype_ = inputs[kParamIndex]->GetDtype();
  auto op_ptr = std::dynamic_pointer_cast<ops::FusedAdaFactor>(base_operator);
  MS_EXCEPTION_IF_NULL(op_ptr);
  enable_scale_parameter_ = op_ptr->get_enable_scale_parameter();
  enable_first_moment_ = op_ptr->get_enable_first_moment();
  enable_weight_decay_ = op_ptr->get_enable_weight_decay();
  return true;
}

int FusedAdaFactorCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }

  auto shape = inputs[kParamIndex]->GetShapeVector();
  elem_num_ = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the elem num of 'param' can not be zero.";
  }
  if (shape.size() >= kLastColIndex) {
    need_factor_ = true;
    last_row_dim_size_ = LongToSize(shape[shape.size() - kLastRowIndex]);
    last_col_dim_size_ = LongToSize(shape[shape.size() - kLastColIndex]);
    if (last_row_dim_size_ < 1 || last_col_dim_size_ < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'param' can not be zero.";
    }
  }

  workspace_size_list_.clear();
  (void)workspace_size_list_.emplace_back(elem_num_ * kSizeFloat32);
  (void)workspace_size_list_.emplace_back((elem_num_ / last_row_dim_size_) * kSizeFloat32);
  (void)workspace_size_list_.emplace_back((elem_num_ / last_col_dim_size_) * kSizeFloat32);
  return KRET_OK;
}

template <typename T>
float FusedAdaFactorCpuKernelMod::CalcRMS(const T *input, size_t elem_num) const {
  if (elem_num == 0 || input == nullptr) {
    return 0.0f;
  }
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  size_t thread_num =
    elem_num < kBatchSize * max_thread_num ? (elem_num + kBatchSize - 1) / kBatchSize : max_thread_num;
  std::vector<common::Task> tasks;
  size_t batch_size = (elem_num + thread_num - 1) / thread_num;
  std::vector<float> block_sum(thread_num, 0.0f);
  for (size_t thread_id = 0; thread_id < thread_num; ++thread_id) {
    size_t start = batch_size * thread_id;
    size_t end = (start + batch_size) > elem_num ? elem_num : (start + batch_size);
    auto block = [&, start, end, thread_id]() {
      float square_sum = 0;
      for (size_t i = start; i < end; ++i) {
        auto tmp = static_cast<float>(input[i]);
        square_sum += tmp * tmp;
      }
      block_sum[thread_id] = square_sum;
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(block);
  }
  ParallelLaunch(tasks);
  auto rms = std::accumulate(block_sum.begin(), block_sum.end(), 0.0f);
  rms = rms / elem_num;
  return std::sqrt(rms);
}

template <typename T>
void FusedAdaFactorCpuKernelMod::FactorUpdate(float *update, const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspaces) const {
  auto beta2t = reinterpret_cast<float *>(inputs[kBeta2tIndex]->addr)[kScalarIndex];
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  auto exp_avg_sq_row = reinterpret_cast<T *>(inputs[kExpAvgSQRowIndex]->addr);
  auto exp_avg_sq_col = reinterpret_cast<T *>(inputs[kExpAvgSQColIndex]->addr);
  auto r_factor = reinterpret_cast<float *>(workspaces[kWorkSpaceRFactorIndex]->addr);
  auto c_factor = reinterpret_cast<float *>(workspaces[kWorkSpaceCFactorIndex]->addr);
  auto one_minus_beta2t = 1 - beta2t;

  std::function<void(size_t, size_t)> task;
  size_t exp_avg_sq_row_elem_num = elem_num_ / last_row_dim_size_;
  size_t exp_avg_sq_col_elem_num = elem_num_ / last_col_dim_size_;
  size_t last_row_col_size = last_row_dim_size_ * last_col_dim_size_;
  size_t row_dim_size = last_row_dim_size_;
  size_t col_dim_size = last_col_dim_size_;
  // calc exp_avg_sq_row
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float row_reduce = 0;
      size_t reduce_start = i * row_dim_size;
      for (size_t j = 0; j < row_dim_size; ++j) {
        row_reduce += update[reduce_start + j];
      }
      row_reduce = row_reduce / row_dim_size;
      auto tmp = static_cast<float>(exp_avg_sq_row[i]) * beta2t + row_reduce * one_minus_beta2t;
      exp_avg_sq_row[i] = static_cast<T>(tmp);
    }
  };
  CPUKernelUtils::ParallelFor(task, exp_avg_sq_row_elem_num, kBatchSize);

  // calc r_factor
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float col_reduce = 0;
      size_t reduce_start = i * col_dim_size;
      for (size_t j = 0; j < col_dim_size; ++j) {
        col_reduce += static_cast<float>(exp_avg_sq_row[reduce_start + j]);
      }
      col_reduce = col_reduce / col_dim_size;
      col_reduce = std::max(col_reduce, kEps);
      for (size_t j = 0; j < col_dim_size; ++j) {
        r_factor[reduce_start + j] = std::sqrt(static_cast<float>(exp_avg_sq_row[reduce_start + j]) / col_reduce);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, exp_avg_sq_row_elem_num / col_dim_size, kBatchSize);

  // calc exp_avg_sq_col and c_factor
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float row_reduce = 0;
      size_t reduce_start = (i / row_dim_size) * last_row_col_size + i % row_dim_size;
      for (size_t j = 0; j < col_dim_size; ++j) {
        row_reduce += update[reduce_start + j * row_dim_size];
      }
      row_reduce = row_reduce / col_dim_size;
      auto tmp = static_cast<float>(exp_avg_sq_col[i]) * beta2t + row_reduce * one_minus_beta2t;
      tmp = std::max(tmp, kEps);
      exp_avg_sq_col[i] = static_cast<T>(tmp);
      c_factor[i] = std::sqrt(tmp);
    }
  };
  CPUKernelUtils::ParallelFor(task, exp_avg_sq_col_elem_num, kBatchSize);

  // calc update
  task = [&, this](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t row_i = i % row_dim_size;
      size_t col_i = i / row_dim_size % col_dim_size;
      size_t slice = i / last_row_col_size;
      auto norm = r_factor[slice * col_dim_size + col_i] * c_factor[slice * row_dim_size + row_i];
      update[i] = static_cast<float>(grad[i]) * global_norm_reciprocal_ / std::max(norm, kEps);
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num_, kBatchSize);
}

template <typename T>
void FusedAdaFactorCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspaces,
                                              const std::vector<AddressPtr> &) {
  auto epsilon = reinterpret_cast<float *>(inputs[kEpsIndex]->addr);
  auto clip_threshold = reinterpret_cast<float *>(inputs[kClipThresholdIndex]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[kBeta1Index]->addr)[kScalarIndex];
  auto beta2t = reinterpret_cast<float *>(inputs[kBeta2tIndex]->addr)[kScalarIndex];
  auto weight_decay = reinterpret_cast<float *>(inputs[kWeightDecayIndex]->addr)[kScalarIndex];
  auto learning_rate = reinterpret_cast<float *>(inputs[kLearningRateIndex]->addr)[kScalarIndex];
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  auto param = reinterpret_cast<T *>(inputs[kParamIndex]->addr);
  auto exp_avg = reinterpret_cast<T *>(inputs[kExpAvgIndex]->addr);
  auto exp_avg_sq = reinterpret_cast<T *>(inputs[kExpAvgSQIndex]->addr);
  auto update = reinterpret_cast<float *>(workspaces[kWorkSpaceUpdateIndex]->addr);
  auto one_minus_beta1 = 1 - beta1;
  auto one_minus_beta2t = 1 - beta2t;
  if (clip_threshold <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', clip threshold " << clip_threshold << " is invalid. ";
  }
  if (beta1 < 0 || one_minus_beta1 < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', beta1 " << beta1 << " is invalid. ";
  }
  if (beta2t < 0 || one_minus_beta2t < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', beta2t " << beta2t << " is invalid. ";
  }
  if (epsilon[0] < 0 || epsilon[1] < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', epsilon (" << epsilon[0] << "," << epsilon[1]
                      << ") is invalid. ";
  }

  std::function<void(size_t, size_t)> task;
  // calc update
  task = [&, this](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      auto tmp = static_cast<float>(grad[i]) * global_norm_reciprocal_;
      update[i] = tmp * tmp + epsilon[0];
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num_, kBatchSize);

  if (need_factor_) {
    FactorUpdate<T>(update, inputs, workspaces);
  } else {
    // no factor
    task = [&, this](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        auto tmp = static_cast<float>(exp_avg_sq[i]) * beta2t + update[i] * one_minus_beta2t;
        tmp = std::max(tmp, kEps);
        exp_avg_sq[i] = static_cast<T>(tmp);
        update[i] = static_cast<float>(grad[i]) * global_norm_reciprocal_ / std::sqrt(tmp);
      }
    };
    CPUKernelUtils::ParallelFor(task, elem_num_, kBatchSize);
  }

  // scale learning rate with rms of param
  if (enable_scale_parameter_) {
    auto rms = CalcRMS(param, elem_num_);
    learning_rate = learning_rate * std::max(epsilon[1], rms);
  }

  // update param
  auto update_rms = CalcRMS(update, elem_num_);
  auto update_rms_threshold = update_rms / clip_threshold;
  auto update_coff = learning_rate / std::max(update_rms_threshold, 1.0f);
  task = [&, this](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      update[i] = update[i] * update_coff;
      if (enable_first_moment_) {
        update[i] = static_cast<float>(exp_avg[i]) * beta1 + update[i] * one_minus_beta1;
        exp_avg[i] = static_cast<T>(update[i]);
      }
      if (enable_weight_decay_) {
        auto tmp = update[i] + static_cast<float>(param[i]) * weight_decay * learning_rate;
        param[i] = static_cast<T>(static_cast<float>(param[i]) - tmp);
      } else {
        param[i] = static_cast<T>(static_cast<float>(param[i]) - update[i]);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num_, kBatchSize);
}

bool FusedAdaFactorCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspaces,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() == kStandardInputNum + 1) {
    auto global_norm = reinterpret_cast<float *>(inputs[kGlobalNormIndex]->addr)[kScalarIndex];
    if (global_norm < kEps) {
      global_norm_reciprocal_ = 1.0f;
    } else {
      global_norm_reciprocal_ = 1.0f / global_norm;
    }
  }

  CheckInputAddresses(inputs);
  CheckWorkspaceAddresses(workspaces);
  if (param_dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspaces, outputs);
  } else {
    LaunchKernel<float>(inputs, workspaces, outputs);
  }
  return true;
}

void FusedAdaFactorCpuKernelMod::CheckInputAddresses(const std::vector<kernel::AddressPtr> &inputs) const {
  if (inputs.size() < kStandardInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be at least " << kStandardInputNum
                      << ", but got: " << inputs.size();
  }

  if (inputs[kEpsIndex]->size != kSizeFloat32 << 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'epsilon' must be " << (kSizeFloat32 << 1)
                      << ", but got " << inputs[kEpsIndex]->size;
  }
  if (inputs[kClipThresholdIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'clip_threshold' must be " << kSizeFloat32
                      << ", but got " << inputs[kClipThresholdIndex]->size;
  }
  if (inputs[kBeta1Index]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'beta1' must be " << kSizeFloat32
                      << ", but got " << inputs[kBeta1Index]->size;
  }
  if (inputs[kBeta2tIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'beta2t' must be " << kSizeFloat32
                      << ", but got " << inputs[kBeta2tIndex]->size;
  }
  if (inputs[kWeightDecayIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'weight_decay' must be " << kSizeFloat32
                      << ", but got " << inputs[kWeightDecayIndex]->size;
  }
  if (inputs[kLearningRateIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'lr' must be " << kSizeFloat32
                      << ", but got " << inputs[kLearningRateIndex]->size;
  }

  size_t param_size = param_dtype_ == kNumberTypeFloat16 ? elem_num_ * kSizeFloat16 : elem_num_ * kSizeFloat32;
  if (inputs[kParamIndex]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'param' must be " << param_size
                      << ", but got " << inputs[kParamIndex]->size;
  }
  if (inputs[kGradIndex]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'gradient' must be " << param_size
                      << ", but got " << inputs[kGradIndex]->size;
  }

  if (enable_first_moment_ && inputs[kExpAvgIndex]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg' must be " << param_size
                      << ", but got " << inputs[kExpAvgIndex]->size;
  }

  if (!need_factor_) {
    if (inputs[kExpAvgSQIndex]->size != param_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq' must be " << param_size
                        << ", but got " << inputs[kExpAvgSQIndex]->size;
    }
    return;
  }

  if (inputs[kExpAvgSQRowIndex]->size != param_size / last_row_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq_row' must be "
                      << param_size / last_row_dim_size_ << ", but got " << inputs[kExpAvgSQRowIndex]->size;
  }
  if (inputs[kExpAvgSQColIndex]->size != param_size / last_col_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq_col' must be "
                      << param_size / last_col_dim_size_ << ", but got " << inputs[kExpAvgSQColIndex]->size;
  }
}

void FusedAdaFactorCpuKernelMod::CheckWorkspaceAddresses(const std::vector<kernel::AddressPtr> &workspaces) const {
  if (workspaces.size() != kWorkSpaceNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of workspaces must be " << kWorkSpaceNum
                      << ", but got: " << workspaces.size();
  }

  size_t update_size = elem_num_ * kSizeFloat32;
  if (workspaces[kWorkSpaceUpdateIndex]->size != elem_num_ * kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'update ' must be " << update_size
                      << ", but got " << workspaces[kWorkSpaceUpdateIndex]->size;
  }

  if (workspaces[kWorkSpaceRFactorIndex]->size != update_size / last_row_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'r_factor' must be "
                      << update_size / last_row_dim_size_ << ", but got " << workspaces[kWorkSpaceRFactorIndex]->size;
  }
  if (workspaces[kWorkSpaceCFactorIndex]->size != update_size / last_col_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'c_factor' must be "
                      << update_size / last_col_dim_size_ << ", but got " << workspaces[kWorkSpaceCFactorIndex]->size;
  }
}

std::vector<KernelAttr> FusedAdaFactorCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {{kFusedAdaFactor,
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
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
                                                                              KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddOutputAttr(kNumberTypeFloat16)}},
                                                                            {kFusedAdaFactorWithGlobalNorm,
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
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
                                                                              KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddOutputAttr(kNumberTypeFloat16)}}};
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }

  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FusedAdaFactor,
                                 []() { return std::make_shared<FusedAdaFactorCpuKernelMod>(kFusedAdaFactor); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FusedAdaFactorWithGlobalNorm, []() {
  return std::make_shared<FusedAdaFactorCpuKernelMod>(kFusedAdaFactorWithGlobalNorm);
});
}  // namespace kernel
}  // namespace mindspore
