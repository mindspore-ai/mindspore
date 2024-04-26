/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/lstsqv2_grad_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/lstsqv2_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
using Eigen::ColMajor;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
constexpr size_t kLstsqV2GradInputsNum = 3;
constexpr size_t kLstsqV2GradOutputsNum = 2;
constexpr size_t kIndexGX = 0;
constexpr size_t kIndexA = 1;
constexpr size_t kIndexB = 2;
constexpr size_t kIndexGA = 0;
constexpr size_t kIndexGB = 1;
constexpr size_t kIndexPinvA = 0;
constexpr size_t kIndexGPinvA = 1;
constexpr size_t kIndexGATemp = 2;
constexpr size_t kIndexGBTemp = 3;
constexpr size_t kMatrixSize = 2;
constexpr size_t kVectorSize = 1;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool LstsqV2GradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLstsqV2GradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLstsqV2GradOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndexGA).dtype);
  return true;
}

int LstsqV2GradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto gx_shape = inputs.at(kIndexGX)->GetShapeVector();
  auto a_shape = inputs.at(kIndexA)->GetShapeVector();
  auto b_shape = inputs.at(kIndexB)->GetShapeVector();
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  size_t batch_dims = a_dims - 2;
  batch_ = 1;
  ga_size_ = 1;
  gb_size_ = 1;
  for (size_t idx = 0; idx < batch_dims; idx++) {
    a_batch_shape_.emplace_back(a_shape[idx]);
    b_batch_shape_.emplace_back(b_shape[idx]);
    broadcast_batch_shape_.emplace_back(gx_shape[idx]);
    batch_ = batch_ * LongToSize(gx_shape[idx]);
    ga_size_ *= LongToSize(a_shape[idx]);
    gb_size_ *= LongToSize(b_shape[idx]);
  }

  m_ = LongToSize(a_shape[a_dims - 2]);
  n_ = LongToSize(a_shape[a_dims - 1]);
  k_ = a_dims == b_dims ? LongToSize(b_shape[b_dims - 1]) : 1;
  a_mat_size_ = m_ * n_;
  b_mat_size_ = m_ * k_;
  ga_size_ *= a_mat_size_;
  gb_size_ *= b_mat_size_;
  gx_mat_size_ = n_ * k_;
  (void)workspace_size_list_.emplace_back(a_mat_size_ * data_unit_size_);
  (void)workspace_size_list_.emplace_back(a_mat_size_ * data_unit_size_);
  (void)workspace_size_list_.emplace_back(a_mat_size_ * data_unit_size_);
  (void)workspace_size_list_.emplace_back(b_mat_size_ * data_unit_size_);
  return KRET_OK;
}

template <typename T>
void Pinv(T *input_addr, T *output_addr, size_t row, size_t col) {
  Map<Matrix<T, RowMajor>> in(input_addr, row, col);
  Map<Matrix<T, RowMajor>> out(output_addr, col, row);
  auto svd = in.bdcSvd(ComputeFullU | ComputeFullV);
  Matrix<T, RowMajor> s = svd.singularValues();
  Matrix<T, RowMajor> s_inv(col, row);
  s_inv.setZero();
  size_t s_size = row < col ? row : col;
  for (size_t i = 0; i < s_size; i++) {
    s_inv(i, i) = static_cast<T>(1) / s(i);
  }
  Map<Matrix<T, RowMajor>>(output_addr, col, row) = svd.matrixV() * s_inv * svd.matrixU().transpose().conjugate();
}

template <typename T>
void pinv_backward(T *g_pinv_addr, T *pinv_a_addr, T *a_addr, T *g_a_addr, size_t row, size_t col) {
  Map<Matrix<T, RowMajor>> g_pinv_a(g_pinv_addr, col, row);
  Map<Matrix<T, RowMajor>> pinv_a(pinv_a_addr, col, row);
  Map<Matrix<T, RowMajor>> a(a_addr, row, col);
  Matrix<T, RowMajor> pinv_a_h = pinv_a.transpose().conjugate();
  Matrix<T, RowMajor> g_pinv_a_h = g_pinv_a.transpose().conjugate();
  if (row <= col) {
    Matrix<T, RowMajor> K = g_pinv_a_h * pinv_a;
    Matrix<T, RowMajor> K_pinv_a_h = K * pinv_a_h;
    Map<Matrix<T, RowMajor>>(g_a_addr, row, col) = -(pinv_a * K).transpose().conjugate() + K_pinv_a_h -
                                                   (a * pinv_a) * K_pinv_a_h +
                                                   (pinv_a_h * pinv_a) * (g_pinv_a_h - K * a);
  } else {
    Matrix<T, RowMajor> K = pinv_a * g_pinv_a_h;
    Matrix<T, RowMajor> pinv_a_h_K = pinv_a_h * K;
    Map<Matrix<T, RowMajor>>(g_a_addr, row, col) = -(K * pinv_a).transpose().conjugate() +
                                                   (g_pinv_a_h - a * K) * pinv_a * pinv_a_h + pinv_a_h_K -
                                                   pinv_a_h_K * pinv_a * a;
  }
}

template <typename T>
void memadd(T *source_addr, T *target_addr, size_t len) {
  for (size_t i = 0; i < len; i++) {
    target_addr[i] += source_addr[i];
  }
}

template <typename T>
bool LstsqV2GradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &workspace,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  if (ga_size_ == 0 || gb_size_ == 0) return true;
  T *input_gx_addr = reinterpret_cast<T *>(inputs[kIndexGX]->device_ptr());
  T *input_a_addr = reinterpret_cast<T *>(inputs[kIndexA]->device_ptr());
  T *input_b_addr = reinterpret_cast<T *>(inputs[kIndexB]->device_ptr());
  T *input_ga_addr = reinterpret_cast<T *>(outputs[kIndexGA]->device_ptr());
  T *input_gb_addr = reinterpret_cast<T *>(outputs[kIndexGB]->device_ptr());

  T *pinv_a_addr = reinterpret_cast<T *>(workspace[kIndexPinvA]->device_ptr());
  T *g_pinv_a_addr = reinterpret_cast<T *>(workspace[kIndexGPinvA]->device_ptr());
  T *ga_temp_addr = reinterpret_cast<T *>(workspace[kIndexGATemp]->device_ptr());
  T *gb_temp_addr = reinterpret_cast<T *>(workspace[kIndexGBTemp]->device_ptr());

  int ret = memset_s(input_ga_addr, outputs[kIndexGA]->size(), 0, sizeof(T) * ga_size_);
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memset_s[input_ga_addr] error. Error no: " << ret;
  }
  ret = memset_s(input_gb_addr, outputs[kIndexGB]->size(), 0, sizeof(T) * gb_size_);
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memset_s[input_gb_addr] error. Error no: " << ret;
  }
  BroadcastIterator batch_broadcast_iter(a_batch_shape_, b_batch_shape_, broadcast_batch_shape_);
  batch_broadcast_iter.SetPos(0);
  for (size_t i = 0; i < batch_; i++) {
    T *a_batch_addr = input_a_addr + batch_broadcast_iter.GetInputPosA() * a_mat_size_;
    T *b_batch_addr = input_b_addr + batch_broadcast_iter.GetInputPosB() * b_mat_size_;
    T *gx_batch_addr = input_gx_addr + i * gx_mat_size_;
    T *ga_batch_addr = input_ga_addr + batch_broadcast_iter.GetInputPosA() * a_mat_size_;
    T *gb_batch_addr = input_gb_addr + batch_broadcast_iter.GetInputPosB() * b_mat_size_;
    Pinv(a_batch_addr, pinv_a_addr, m_, n_);
    Map<Matrix<T, RowMajor>> gX(gx_batch_addr, n_, k_);
    Map<Matrix<T, RowMajor>> b(b_batch_addr, m_, k_);
    Map<Matrix<T, RowMajor>> pinvA(pinv_a_addr, n_, m_);
    Map<Matrix<T, RowMajor>> gPinvA(g_pinv_a_addr, n_, m_);
    Map<Matrix<T, RowMajor>> gB(gb_temp_addr, m_, k_);
    gPinvA = gX * b.transpose().conjugate();
    gB = pinvA.transpose().conjugate() * gX;
    pinv_backward(g_pinv_a_addr, pinv_a_addr, a_batch_addr, ga_temp_addr, m_, n_);
    memadd(ga_temp_addr, ga_batch_addr, a_mat_size_);
    memadd(gb_temp_addr, gb_batch_addr, b_mat_size_);
    batch_broadcast_iter.GenNextPos();
  }
  return true;
}

#define LSTSQV2_GRAD_CPU_REG(T1, T2) \
  KernelAttr()                       \
    .AddInputAttr(T1)   /* gX */     \
    .AddInputAttr(T1)   /* A */      \
    .AddInputAttr(T1)   /* B */      \
    .AddOutputAttr(T1)  /* gA */     \
    .AddOutputAttr(T1), /* gB */     \
    &LstsqV2GradCpuKernelMod::LaunchKernel<T2>

std::vector<std::pair<KernelAttr, LstsqV2GradCpuKernelMod::LstsqV2GradFunc>> LstsqV2GradCpuKernelMod::func_list_ = {
  {LSTSQV2_GRAD_CPU_REG(kNumberTypeFloat32, float)},
  {LSTSQV2_GRAD_CPU_REG(kNumberTypeFloat64, double)},
  {LSTSQV2_GRAD_CPU_REG(kNumberTypeComplex64, complex64)},
  {LSTSQV2_GRAD_CPU_REG(kNumberTypeComplex128, complex128)},
};

std::vector<KernelAttr> LstsqV2GradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LstsqV2GradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LstsqV2Grad, LstsqV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
