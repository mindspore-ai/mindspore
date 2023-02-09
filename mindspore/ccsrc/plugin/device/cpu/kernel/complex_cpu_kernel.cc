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

#include "plugin/device/cpu/kernel/complex_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace {
constexpr size_t kComplexInputsNum = 2;
constexpr size_t kComplexOutputsNum = 1;
constexpr size_t kMaxDims = 8;

template <typename T>
struct ComplexOp {
  typedef std::complex<T> result_type;
  inline result_type operator()(const T &real, const T &image) const { return std::complex<T>(real, image); }
};

bool NeedBroadcast(const std::vector<int64_t> &shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] != 1) {
      return true;
    }
  }
  return false;
}
}  // namespace

namespace mindspore {
namespace kernel {
bool ComplexCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  return true;
}

int ComplexCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  real_shape_ = inputs[0]->GetShapeVector();
  image_shape_ = inputs[1]->GetShapeVector();
  out_shape_ = outputs[0]->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(real_shape_, kernel_name_, "real") ||
                   CHECK_SHAPE_NULL(image_shape_, kernel_name_, "image") ||
                   CHECK_SHAPE_NULL(out_shape_, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  real_bcast_.clear();
  image_bcast_.clear();
  auto real_offset = out_shape_.size() - real_shape_.size();
  auto image_offset = out_shape_.size() - image_shape_.size();
  for (size_t i = 0; i < out_shape_.size(); i++) {
    if (i < real_offset) {
      real_bcast_.push_back(out_shape_[i]);
    } else {
      real_bcast_.push_back(out_shape_[i] / real_shape_[i - real_offset]);
    }

    if (i < image_offset) {
      image_bcast_.push_back(out_shape_[i]);
    } else {
      image_bcast_.push_back(out_shape_[i] / image_shape_[i - image_offset]);
    }
  }

  return KRET_OK;
}

template <typename T>
bool ComplexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }

  auto real_need_broadcast = NeedBroadcast(real_bcast_);
  auto image_need_broadcast = NeedBroadcast(image_bcast_);
  if (!real_need_broadcast && !image_need_broadcast) {
    int64_t num = std::accumulate(out_shape_.begin(), out_shape_.end(), 1, std::multiplies<int64_t>());
    ShapeVector shape = {num};
    auto real = EigenTensor(shape, inputs[kIndex0]->addr).tensor<T, 1>();
    auto image = EigenTensor(shape, inputs[kIndex1]->addr).tensor<T, 1>();
    auto out = EigenTensor(shape, outputs[kIndex0]->addr).tensor<std::complex<T>, 1>();
    // cppcheck-suppress *
    out = real.binaryExpr(image, ComplexOp<T>());
    return true;
  }

  auto real_shape = EigenTensor(real_bcast_, nullptr).AsEigenDSizes<kMaxDims>();
  auto real = EigenTensor(real_shape_, inputs[kIndex0]->addr).tensor<T, kMaxDims>().broadcast(real_shape);
  auto image_shape = EigenTensor(image_bcast_, nullptr).AsEigenDSizes<kMaxDims>();
  auto image = EigenTensor(image_shape_, inputs[kIndex1]->addr).tensor<T, kMaxDims>().broadcast(image_shape);
  auto out = EigenTensor(out_shape_, outputs[kIndex0]->addr).tensor<std::complex<T>, kMaxDims>();
  // cppcheck-suppress *
  out = real.binaryExpr(image, ComplexOp<T>());
  return true;
}

std::vector<std::pair<KernelAttr, ComplexCpuKernelMod::ComplexLaunchFunc>> ComplexCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
   &ComplexCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
   &ComplexCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ComplexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ComplexCpuKernelMod::ComplexLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Complex, ComplexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
