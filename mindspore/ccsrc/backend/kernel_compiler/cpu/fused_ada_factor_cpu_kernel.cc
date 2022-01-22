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
#include "backend/kernel_compiler/cpu/fused_ada_factor_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
static constexpr size_t kSizeFloat32 = sizeof(float);
static constexpr size_t kSizeFloat16 = sizeof(float16);
static constexpr size_t kScalarIndex = 0;
static constexpr size_t kStandardInputNum = 12;
static constexpr size_t kWorkSpaceNum = 3;
static constexpr size_t kBatchSize = 10000;
static auto constexpr kEnableScaleParameter = "enable_scale_parameter";
static auto constexpr kEnableFirstMoment = "enable_first_moment";
static auto constexpr kEnableWeightDecay = "enable_weight_decay";
static constexpr size_t kLastRowIndex = 1;
static constexpr size_t kLastColIndex = 2;
static constexpr float kEps = 1e-30;
}  // namespace

void FusedAdaFactorCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  (void)workspace_size_list_.emplace_back(elem_num_ * kSizeFloat32);
  (void)workspace_size_list_.emplace_back(elem_num_ / last_row_dim_size_ * kSizeFloat32);
  (void)workspace_size_list_.emplace_back(elem_num_ / last_col_dim_size_ * kSizeFloat32);
}

void FusedAdaFactorCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  param_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, PARAM);
  auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, PARAM);
  elem_num_ = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<size_t>());
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the elem num of 'param' should not be zero.";
  }
  if (shape.size() >= kLastColIndex) {
    need_factor_ = true;
    last_row_dim_size_ = shape[shape.size() - kLastRowIndex];
    last_col_dim_size_ = shape[shape.size() - kLastColIndex];
    if (last_row_dim_size_ < 1 || last_col_dim_size_ < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'param' should not be zero.";
    }
  }

  if (AnfAlgo::HasNodeAttr(kEnableScaleParameter, kernel_node)) {
    enable_scale_parameter_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, kEnableScaleParameter);
  }
  if (AnfAlgo::HasNodeAttr(kEnableFirstMoment, kernel_node)) {
    enable_first_moment_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, kEnableFirstMoment);
  }
  if (AnfAlgo::HasNodeAttr(kEnableWeightDecay, kernel_node)) {
    enable_weight_decay_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, kEnableWeightDecay);
  }
}

template <typename T>
float FusedAdaFactorCpuKernelMod::CalcRMS(T *input, size_t elem_num) {
  if (elem_num == 0) {
    return 0.0f;
  }

  float rms = 0;
  for (size_t i = 0; i < elem_num; ++i) {
    auto tmp = static_cast<float>(input[i]);
    rms += tmp * tmp;
  }
  rms /= elem_num;
  return std::sqrt(rms);
}

template <typename T>
void FusedAdaFactorCpuKernelMod::FactorUpdate(float *update, const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspaces) {
  auto beta2t = reinterpret_cast<float *>(inputs[BETA2T]->addr)[kScalarIndex];
  auto grad = reinterpret_cast<T *>(inputs[GRAD]->addr);
  auto exp_avg_sq_row = reinterpret_cast<T *>(inputs[EXP_AVG_SQ_ROW]->addr);
  auto exp_avg_sq_col = reinterpret_cast<T *>(inputs[EXP_AVG_SQ_COL]->addr);
  auto r_factor = reinterpret_cast<float *>(workspaces[R_FACTOR]->addr);
  auto c_factor = reinterpret_cast<float *>(workspaces[C_FACTOR]->addr);
  auto one_minus_beta2t = 1 - beta2t;

  std::function<void(size_t, size_t)> task;
  size_t exp_avg_sq_row_elem_num = elem_num_ / last_row_dim_size_;
  size_t exp_avg_sq_col_elem_num = elem_num_ / last_col_dim_size_;
  size_t last_row_col_size = last_row_dim_size_ * last_col_dim_size_;
  size_t row_dim_size = last_row_dim_size_;
  size_t col_dim_size = last_col_dim_size_;
  // exp_avg_sq_row = exp_avg_sq_row * beta2t + reduce_mean(update, -1) * one_minus_beta2t;
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

  // r_factor = sqrt(exp_avg_sq_row / reduce_mean(exp_avg_sq_row, -1))
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float col_reduce = 0;
      size_t reduce_start = i * col_dim_size;
      for (size_t j = 0; j < col_dim_size; ++j) {
        col_reduce += static_cast<float>(exp_avg_sq_row[reduce_start + j]);
      }
      col_reduce /= col_dim_size;
      col_reduce = std::max(col_reduce, kEps);
      for (size_t j = 0; j < col_dim_size; ++j) {
        r_factor[reduce_start + j] = std::sqrt(static_cast<float>(exp_avg_sq_row[reduce_start + j]) / col_reduce);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, exp_avg_sq_row_elem_num / col_dim_size, kBatchSize);

  // exp_avg_sq_col = exp_avg_sq_col * beta2t + reduce_mean(update, -2) * one_minus_beta2t;
  // c_factor = sqrt(exp_avg_sq_col);
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float row_reduce = 0;
      size_t reduce_start = i / row_dim_size * last_row_col_size + i % row_dim_size;
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

  // update = grad / (r_factor * c_factor);
  task = [&](size_t start, size_t end) {
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
  auto epsilon = reinterpret_cast<float *>(inputs[EPSILON]->addr);
  auto clip_threshold = reinterpret_cast<float *>(inputs[CLIP_THRESHOLD]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2t = reinterpret_cast<float *>(inputs[BETA2T]->addr)[kScalarIndex];
  auto weight_decay = reinterpret_cast<float *>(inputs[WEIGHT_DECAY]->addr)[kScalarIndex];
  auto learning_rate = reinterpret_cast<float *>(inputs[LEARNING_RATE]->addr)[kScalarIndex];
  auto grad = reinterpret_cast<T *>(inputs[GRAD]->addr);
  auto param = reinterpret_cast<T *>(inputs[PARAM]->addr);
  auto exp_avg = reinterpret_cast<T *>(inputs[EXP_AVG]->addr);
  auto exp_avg_sq = reinterpret_cast<T *>(inputs[EXP_AVG_SQ]->addr);
  auto update = reinterpret_cast<float *>(workspaces[UPDATE]->addr);
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
  // update = grad * grad + eps[0]
  task = [&](size_t start, size_t end) {
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
    task = [&](size_t start, size_t end) {
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
  auto update_rms_thres = CalcRMS(update, elem_num_) / clip_threshold;
  auto update_coff = learning_rate / std::max(update_rms_thres, 1.0f);
  task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      update[i] = update[i] * update_coff;
      if (enable_first_moment_) {
        update[i] = static_cast<float>(exp_avg[i]) * beta1 + update[i] * one_minus_beta1;
        exp_avg[i] = static_cast<T>(update[i]);
      }
      if (enable_weight_decay_) {
        auto tmp = static_cast<float>(param[i]) * weight_decay * learning_rate;
        param[i] = static_cast<T>(static_cast<float>(param[i]) - update[i] - tmp);
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
    auto global_norm = reinterpret_cast<float *>(inputs[GLOBAL_NORM]->addr)[kScalarIndex];
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
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be at least " << kStandardInputNum
                      << ", but got: " << inputs.size();
  }

  if (inputs[EPSILON]->size != kSizeFloat32 << 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'epsilon' should be " << (kSizeFloat32 << 1)
                      << ", but got " << inputs[EPSILON]->size;
  }
  if (inputs[CLIP_THRESHOLD]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'beta1' should be " << kSizeFloat32
                      << ", but got " << inputs[BETA1]->size;
  }
  if (inputs[BETA1]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'beta1' should be " << kSizeFloat32
                      << ", but got " << inputs[BETA1]->size;
  }
  if (inputs[BETA2T]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'beta2t' should be " << kSizeFloat32
                      << ", but got " << inputs[BETA2T]->size;
  }
  if (inputs[WEIGHT_DECAY]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'weight_decay' should be " << kSizeFloat32
                      << ", but got " << inputs[WEIGHT_DECAY]->size;
  }
  if (inputs[LEARNING_RATE]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'lr' should be " << kSizeFloat32
                      << ", but got " << inputs[LEARNING_RATE]->size;
  }

  size_t param_size = param_dtype_ == kNumberTypeFloat16 ? elem_num_ * kSizeFloat16 : elem_num_ * kSizeFloat32;
  if (inputs[PARAM]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'param' should be " << param_size
                      << ", but got " << inputs[PARAM]->size;
  }
  if (inputs[GRAD]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'gradient' should be " << param_size
                      << ", but got " << inputs[GRAD]->size;
  }

  if (enable_first_moment_ && inputs[EXP_AVG]->size != param_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg' should be " << param_size
                      << ", but got " << inputs[EXP_AVG]->size;
  }

  if (!need_factor_) {
    if (inputs[EXP_AVG_SQ]->size != param_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq' should be " << param_size
                        << ", but got " << inputs[EXP_AVG_SQ]->size;
    }
    return;
  }

  if (inputs[EXP_AVG_SQ_ROW]->size != param_size / last_row_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq_row' should be "
                      << param_size / last_row_dim_size_ << ", but got " << inputs[EXP_AVG_SQ_ROW]->size;
  }
  if (inputs[EXP_AVG_SQ_COL]->size != param_size / last_col_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'exp_avg_sq_col' should be "
                      << param_size / last_col_dim_size_ << ", but got " << inputs[EXP_AVG_SQ_COL]->size;
  }
}

void FusedAdaFactorCpuKernelMod::CheckWorkspaceAddresses(const std::vector<kernel::AddressPtr> &workspaces) const {
  if (workspaces.size() != kWorkSpaceNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of workspaces should be " << kWorkSpaceNum
                      << ", but got: " << workspaces.size();
  }

  size_t update_size = elem_num_ * kSizeFloat32;

  if (workspaces[UPDATE]->size != elem_num_ * kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'update ' should be " << update_size
                      << ", but got " << workspaces[0]->size;
  }

  if (workspaces[R_FACTOR]->size != update_size / last_row_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'r_factor' should be "
                      << update_size / last_row_dim_size_ << ", but got " << workspaces[R_FACTOR]->size;
  }
  if (workspaces[C_FACTOR]->size != update_size / last_col_dim_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'c_factor' should be "
                      << update_size / last_col_dim_size_ << ", but got " << workspaces[C_FACTOR]->size;
  }
}
}  // namespace kernel
}  // namespace mindspore
