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

#include "plugin/device/cpu/kernel/ctcloss_v2_grad_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <string>
#include <cmath>
#include "abstract/utils.h"
#include "mindspore/core/ops/ctc_loss_v2_grad.h"
namespace mindspore {
namespace kernel {
namespace {
#define CTC_GRAD_KERNEL std::vector<std::pair<KernelAttr, CTCLossV2GradCpuKernelMod::KernelRunFunc>>

constexpr int64_t target_mul = 2;
struct SoftParam {
  int64_t blank_;
  int64_t input_length;
  int64_t target_length;
  int64_t offset;
  int64_t batch;
  void *targets;
};

template <typename target_t>
static inline int64_t get_target_prime(const target_t *target, int64_t offset, int64_t idx, int64_t BLANK) {
  constexpr int64_t even = 2;
  if (idx % even == 0) {
    return BLANK;
  } else {
    return target[offset + (idx / even)];
  }
}
}  // namespace
bool CTCLossV2GradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  // Getting values
  auto kernel_ptr = std::make_shared<ops::CTCLossV2Grad>(base_operator->GetPrim());
  blank_ = kernel_ptr->get_blank();
  zero_infinity_ = kernel_ptr->get_zero_infinity();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int CTCLossV2GradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto log_probs_shape = inputs[kIndex1]->GetShapeVector();
  T_ = log_probs_shape[kIndex0];
  batch_size_ = log_probs_shape[kIndex1];
  num_labels_ = log_probs_shape[kIndex2];
  const auto target_shape = inputs[kIndex2]->GetShapeVector();
  max_target_length_ = target_shape[kIndex1];

  const size_t scalar_type_size = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  workspace_size_list_.clear();
  workspace_size_list_ = {
    LongToSize(batch_size_ * T_ * (target_mul * max_target_length_ + 1)) * scalar_type_size,
  };
  return KRET_OK;
}

template <typename scalar_t, typename target_t>
void ComputeGrad(const scalar_t *log_probs, const NdTensorIterator<kDim3> &log_probs_it, SoftParam params,
                 const scalar_t *log_alpha, const NdTensorIterator<kDim3> &log_alpha_it, scalar_t *log_beta,
                 const NdTensorIterator<kDim3> &log_beta_it, scalar_t *grad, const NdTensorIterator<kDim3> &grad_it) {
  int64_t blank_ = params.blank_;
  int64_t input_length = params.input_length;
  int64_t target_length = params.target_length;
  int64_t tg_batch_offset = params.offset;
  int64_t b = params.batch;
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  target_t *targets = static_cast<target_t *>(params.targets);
  for (int64_t t = input_length - 2; t >= 0; t--) {
    for (int64_t s = 2 * target_length; s >= 0; s--) {
      scalar_t lb1 = log_beta[log_beta_it(b, t + 1, s)];
      scalar_t lbmax = lb1;
      scalar_t lb2, lb3;
      auto current_target_prime = get_target_prime(targets, tg_batch_offset, s, blank_);
      if (s < target_mul * target_length) {
        lb2 = log_beta[log_beta_it(b, t + 1, s + 1)];
        if (lb2 > lbmax) {
          lbmax = lb2;
        }
      } else {
        lb2 = neginf;
      }
      if ((s < target_mul * target_length - 1) &&
          (get_target_prime(targets, tg_batch_offset, s + target_mul, blank_) != current_target_prime)) {
        lb3 = log_beta[log_beta_it(b, t + 1, s + target_mul)];
        if (lb3 > lbmax) {
          lbmax = lb3;
        }
      } else {
        lb3 = neginf;
      }
      if (std::isinf(lbmax) && std::signbit(lbmax)) {
        lbmax = 0;
      }
      log_beta[log_beta_it(b, t, s)] = std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) + std::exp(lb3 - lbmax)) +
                                       lbmax + log_probs[log_probs_it(t, b, current_target_prime)];
      scalar_t log_alpha_beta = log_alpha[log_alpha_it(b, t, s)] + log_beta[log_beta_it(b, t, s)];
      scalar_t &lcab = grad[grad_it(t, b, current_target_prime)];
      if (std::isinf(lcab) && std::signbit(lcab)) {
        lcab = log_alpha_beta;
      } else {
        scalar_t max = std::max(lcab, log_alpha_beta);
        lcab = std::log(std::exp(lcab - max) + std::exp(log_alpha_beta - max)) + max;
      }
    }
  }
}
template <typename scalar_t, typename target_t>
bool CTCLossV2GradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &workspace,
                                             const std::vector<kernel::AddressPtr> &outputs) const {
  auto grad_out = static_cast<scalar_t *>(inputs[kIndex0]->addr);
  auto log_probs = static_cast<scalar_t *>(inputs[kIndex1]->addr);
  auto targets = static_cast<target_t *>(inputs[kIndex2]->addr);
  auto input_lengths = static_cast<target_t *>(inputs[kIndex3]->addr);
  auto target_lengths = static_cast<target_t *>(inputs[kIndex4]->addr);
  auto neg_log_likelihood = static_cast<scalar_t *>(inputs[kIndex5]->addr);
  auto log_alpha = static_cast<scalar_t *>(inputs[kIndex6]->addr);
  auto log_beta = static_cast<scalar_t *>(workspace[kIndex0]->addr);
  auto grad = static_cast<scalar_t *>(outputs[kIndex0]->addr);

  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  std::fill(grad, grad + (T_ * batch_size_ * num_labels_), neginf);
  NdTensorIterator<kDim3> log_probs_it(T_, batch_size_, num_labels_);
  NdTensorIterator<kDim3> grad_it(T_, batch_size_, num_labels_);

  NdTensorIterator<kDim3> log_alpha_it(batch_size_, T_, target_mul * max_target_length_ + 1);
  NdTensorIterator<kDim3> log_beta_it(batch_size_, T_, target_mul * max_target_length_ + 1);
  for (int64_t b = 0; b < batch_size_; b++) {
    scalar_t nll = neg_log_likelihood[b];
    if (zero_infinity_ && std::isinf(nll) && !std::signbit(nll)) {
      for (int t = 0; t < T_; t++) {
        for (int c = 0; c < num_labels_; c++) {
          grad[grad_it(t, b, c)] = 0;
        }
      }
      continue;
    }
    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t tg_batch_offset = max_target_length_ * b;
    if (input_length > 0) {
      for (size_t s = 0; s < LongToSize(target_mul * max_target_length_ + 1); s++) {
        log_beta[log_beta_it(b, input_length - 1, s)] = neginf;
      }
      log_beta[log_beta_it(b, input_length - 1, target_mul * target_length)] =
        log_probs[log_probs_it(input_length - 1, b, blank_)];
      grad[grad_it(input_length - 1, b, blank_)] =
        log_alpha[log_alpha_it(b, input_length - 1, target_mul * target_length)] +
        log_beta[log_beta_it(b, input_length - 1, target_mul * target_length)];
      if (target_length > 0) {
        auto current_target_prime = get_target_prime(targets, tg_batch_offset, target_mul * target_length - 1, blank_);
        log_beta[log_beta_it(b, input_length - 1, target_mul * target_length - 1)] =
          log_probs[log_probs_it(input_length - 1, b, current_target_prime)];
        grad[grad_it(input_length - 1, b, current_target_prime)] =
          log_alpha[log_alpha_it(b, input_length - 1, target_mul * target_length - 1)] +
          log_beta[log_beta_it(b, input_length - 1, target_mul * target_length - 1)];
      }
    }
    SoftParam param = {blank_, input_length, target_length, tg_batch_offset, b, static_cast<void *>(targets)};
    ComputeGrad<scalar_t, target_t>(log_probs, log_probs_it, param, log_alpha, log_alpha_it, log_beta, log_beta_it,
                                    grad, grad_it);
    scalar_t gr = grad_out[b];
    for (int64_t t = 0; t < input_length; t++) {
      for (int64_t c = 0; c < num_labels_; c++) {
        scalar_t &res = grad[grad_it(t, b, c)];
        scalar_t lp = log_probs[log_probs_it(t, b, c)];
        res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
      }
    }
    for (auto l = input_length; l < T_; l++) {
      for (int c = 0; c < num_labels_; c++) {
        grad[grad_it(l, b, c)] = 0;
      }
    }
  }

  return true;
}

const CTC_GRAD_KERNEL &CTCLossV2GradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CTCLossV2GradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GradCpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GradCpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GradCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GradCpuKernelMod::LaunchKernel<double, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CTCLossV2Grad, CTCLossV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
