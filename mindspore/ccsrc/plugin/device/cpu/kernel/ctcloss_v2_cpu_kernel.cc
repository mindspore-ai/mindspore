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

#include "plugin/device/cpu/kernel/ctcloss_v2_cpu_kernel.h"
#include <utility>
#include <map>
#include <string>
#include <limits>
#include <algorithm>
#include "mindspore/core/ops/ctc_loss_v2.h"
#include "abstract/utils.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t target_mul = 2;
}  // namespace
bool CTCLossV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::CTCLossV2>(base_operator->GetPrim());
  blank_ = kernel_ptr->get_blank();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}
int CTCLossV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  const auto log_probs_shape = inputs[kIndex0]->GetShapeVector();
  time_series_ = log_probs_shape[kIndex0];
  batch_sizes_ = log_probs_shape[kIndex1];
  num_labels_ = log_probs_shape[kIndex2];
  const auto target_shape = inputs[kIndex1]->GetShapeVector();
  max_target_length_ = target_shape[kIndex1];
  const auto input_length_shape = inputs[kIndex2]->GetShapeVector();
  if (!(blank_ >= 0 && blank_ < num_labels_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the attr blank must be in label range [ 0, " << num_labels_
                  << " ), but got value " << blank_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_length_shape.size() != 1 || input_length_shape[0] != batch_sizes_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'input_length' must be one-dimensional, "
                     "and the size is equal to batch_size: "
                  << batch_sizes_ << ", but got the shape of 'input_length': " << Vector2Str(input_length_shape) << ".";
    return KRET_RESIZE_FAILED;
  }
  const auto target_length_shape = inputs[kIndex3]->GetShapeVector();
  if (target_length_shape.size() != 1 || target_length_shape[0] != batch_sizes_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'target_length' must be one-dimensional, "
                     "and the size is equal to batch_size: "
                  << batch_sizes_ << ", but got the shape of 'target_length': " << Vector2Str(target_length_shape)
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename S, typename T>
void CTCLossV2CpuKernelMod::LossCompute(const S *log_probs_p, S *log_alpha_p, const T *tar_p, SoftParam params) const {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  int64_t input_length = params.input_length;
  int64_t target_length = params.target_length;
  int64_t offset = params.offset;
  int64_t batch = params.batch;
  NdTensorIterator<kDim3> log_probs_it(time_series_, batch_sizes_, num_labels_);
  NdTensorIterator<kDim3> log_alpha_it(batch_sizes_, time_series_, target_mul * max_target_length_ + 1);
  if (target_length > 0) {
    log_alpha_p[log_alpha_it(batch, 0, 1)] =
      log_probs_p[log_probs_it(0, batch, GetBlankPaddedTarget(tar_p, offset, 1))];
  }
  for (int64_t s = 0; s < target_mul * target_length + 1; s++) {
    auto current_target_prime = GetBlankPaddedTarget(tar_p, offset, s);
    bool three_sum = (s > 1) && (GetBlankPaddedTarget(tar_p, offset, s - target_mul) != current_target_prime);
    // a1 is the result of the previous loop
    S log_a1 = log_alpha_p[log_alpha_it(batch, 0, s)];
    for (int64_t t = 1; t < input_length; t++) {
      S log_max = log_a1;
      S log_a2, log_a3;
      if (s > 0) {
        log_a2 = log_alpha_p[log_alpha_it(batch, t - 1, s - 1)];
        log_max = std::max(log_a2, log_max);
      } else {
        log_a2 = neg_inf;
      }
      if (three_sum) {
        log_a3 = log_alpha_p[log_alpha_it(batch, t - 1, s - target_mul)];
        log_max = std::max(log_a3, log_max);
      } else {
        log_a3 = neg_inf;
      }
      if (std::isinf(log_max) && std::signbit(log_max)) {
        log_max = 0;
      }
      S log_three_sum = std::log(std::exp(log_a1 - log_max) + std::exp(log_a2 - log_max) + std::exp(log_a3 - log_max)) +
                        log_max + log_probs_p[log_probs_it(t, batch, current_target_prime)];
      log_alpha_p[log_alpha_it(batch, t, s)] = log_three_sum;
      log_a1 = log_three_sum;
    }
  }
}

template <typename T>
bool CTCLossV2CpuKernelMod::IndexProcessing(const T *in_len_p, const T *tar_len_p,
                                            std::vector<int64_t> *target_offsets) {
  const int64_t target_stride = max_target_length_;
  for (size_t i = 0; i < LongToSize(batch_sizes_); ++i) {
    if (tar_len_p[i] < 0 || tar_len_p[i] > target_stride) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the target_lengths[" << i << "] = " << tar_len_p[i]
                    << " is negative or larger than target.shape[1] = " << target_stride << ".";
      return false;
    }
    (*target_offsets)[i] = target_stride * SizeToLong(i);
  }
  for (size_t b = 0; b < LongToSize(batch_sizes_); ++b) {
    const auto input_length = in_len_p[b];
    const auto target_length = tar_len_p[b];
    if (input_length > time_series_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input_lengths[" << b << "] = " << input_length
                    << " should be smaller than probs.shape[0] = " << time_series_;
      return false;
    }
    if (input_length < 0 || input_length < target_length) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input_lengths[" << b << "] = " << input_length
                    << " should be non-negative and smaller than tar_len_p[" << b << "] = " << target_length;
      return false;
    }
  }
  return true;
}

template <typename S, typename T>
bool CTCLossV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto log_probs_p = static_cast<S *>(inputs[kIndex0]->addr);
  auto tar_p = static_cast<T *>(inputs[kIndex1]->addr);
  auto in_len_p = static_cast<T *>(inputs[kIndex2]->addr);
  auto tar_len_p = static_cast<T *>(inputs[kIndex3]->addr);
  auto neg_log_p = static_cast<S *>(outputs[kIndex0]->addr);
  auto log_alpha_p = static_cast<S *>(outputs[kIndex1]->addr);
  std::vector<int64_t> target_offsets(LongToSize(batch_sizes_));
  if (!IndexProcessing<T>(in_len_p, tar_len_p, &target_offsets)) {
    return false;
  }
  const int64_t padding_target_length = target_mul * max_target_length_ + 1;
  NdTensorIterator<kDim3> log_probs_it(time_series_, batch_sizes_, num_labels_);
  NdTensorIterator<kDim3> log_alpha_it(batch_sizes_, time_series_, padding_target_length);
  std::fill(log_alpha_p, log_alpha_p + (batch_sizes_ * time_series_ * padding_target_length),
            -std::numeric_limits<S>::infinity());
  auto task = [&](size_t start, size_t end) {
    for (size_t b = start; b < end; b++) {
      int64_t in_len = in_len_p[b];
      int64_t tar_len = tar_len_p[b];
      int64_t offset = target_offsets[b];
      log_alpha_p[log_alpha_it(b, 0, 0)] = log_probs_p[log_probs_it(0, b, blank_)];
      SoftParam param = {in_len, tar_len, offset, SizeToLong(b)};
      LossCompute<S, T>(log_probs_p, log_alpha_p, tar_p, param);
      if (tar_len == 0) {
        neg_log_p[b] = -log_alpha_p[log_alpha_it(b, in_len - 1, 0)];
      } else {
        S l1 = log_alpha_p[log_alpha_it(b, in_len - 1, tar_len * target_mul)];
        S l2 = log_alpha_p[log_alpha_it(b, in_len - 1, tar_len * target_mul - 1)];
        S m = std::max(l1, l2);
        m = ((std::isinf(m) && std::signbit(m)) ? 0 : m);
        S log_likelihood = std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
        neg_log_p[b] = -log_likelihood;
      }
    }
  };
  ParallelLaunchAutoSearch(task, LongToSize(batch_sizes_), this, &parallel_search_info_);
  return true;
}
const std::vector<std::pair<KernelAttr, CTCLossV2CpuKernelMod::KernelRunFunc>> &CTCLossV2CpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, CTCLossV2CpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2CpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2CpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2CpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2CpuKernelMod::LaunchKernel<double, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CTCLossV2, CTCLossV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
