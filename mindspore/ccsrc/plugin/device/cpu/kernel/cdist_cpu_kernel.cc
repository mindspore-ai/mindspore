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

#include "plugin/device/cpu/kernel/cdist_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/cdist_fp32.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCdistInputDimsMin = 2;
void CdistPNormalOptSpecial(const float *a, const float *b, float *dst, int64_t m, float p) {
  /*
  When m = 1, x < 1, p is large, the result of std::pow(x, p)
  is different in graph mode and pynative mode.
  And the result under graph is 0, resulting in precision problems.
  In order to circumvent this situation, special treatment is done when m = 1.
  */
  if (p == static_cast<float>(0.0)) {
    MS_LOG(ERROR) << "Invalid p, p should not be equal to zeor, bug got p =  " << p;
    return;
  }
  if (m == 1) {
    float res = std::abs(a[0] - b[0]);
    *dst = res;
    return;
  }

  float result = 0;
  int64_t i = 0;
  for (; i < m; i++) {
    float x = std::abs(a[i] - b[i]);
    result += std::pow(x, p);
  }

  float r_p = static_cast<float>(1.) / p;
  result = std::pow(result, r_p);
  *dst = result;

  return;
}
const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

void CdistCpuKernelMod::InitFunc(float p) {
  if (p == 0.0) {
    dist_func_ = CdistZeroNormalOpt;
  } else if (p == 1.0) {
    dist_func_ = CdistOneNormalOpt;
  } else if (p == 2.0) {
    dist_func_ = CdistTwoNormalOpt;
  } else if (std::isinf(p)) {
    dist_func_ = CdistInfNormalOpt;
  } else {
    dist_func_ = CdistPNormalOptSpecial;
  }
}

bool CdistCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Cdist>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast Cdist ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  p_ = kernel_ptr->get_p();

  auto input_type_id = inputs[0]->GetDtype();
  switch (input_type_id) {
    case kNumberTypeFloat32:
      InitFunc(p_);
      break;
    default:
      MS_LOG(ERROR) << "cdist kernel does not support " << TypeIdToString(input_type_id);
      return false;
  }
  return true;
}

int CdistCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  std::vector<int64_t> in_shape0 = inputs[0]->GetShapeVector();
  std::vector<int64_t> in_shape1 = inputs[1]->GetShapeVector();
  auto in_shape_size = in_shape0.size();
  if (in_shape1.size() != in_shape_size || in_shape_size < kCdistInputDimsMin) {
    MS_LOG(ERROR) << "invalid input shape, input0 shape size " << in_shape_size << ", input1 shape size "
                  << in_shape1.size() << ", kernel_name_ " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }
  batch_ = 1;
  for (size_t i = 0; i < in_shape_size - kCdistInputDimsMin; i++) {
    batch_ *= in_shape0[i];
  }

  r0_ = in_shape0[in_shape_size - 2];
  m_ = in_shape0[in_shape_size - 1];
  r1_ = in_shape1[in_shape_size - 2];

  return 0;
}

bool CdistCpuKernelMod::LaunchKernel(int64_t start, int64_t end) {
  const auto *in_data0 = reinterpret_cast<float *>(in_data0_) + start * r0_ * m_;
  const auto *in_data1 = reinterpret_cast<float *>(in_data1_) + start * r1_ * m_;
  auto *out_data = reinterpret_cast<float *>(out_data_) + start * r0_ * r1_;

  for (int64_t b_i = 0; b_i < end - start; b_i++) {
    for (int64_t p_i = 0; p_i < r0_; p_i++) {
      auto in_data_tmp1 = in_data1;
      for (int64_t r_i = 0; r_i < r1_; r_i++) {
        dist_func_(in_data0, in_data_tmp1, &(out_data[r_i]), m_, p_);
        in_data_tmp1 = in_data_tmp1 + m_;
      }
      in_data0 = in_data0 + m_;
      out_data = out_data + r1_;
    }
    in_data1 = in_data1 + r1_ * m_;
  }

  return true;
}

std::vector<KernelAttr> CdistCpuKernelMod::GetOpSupport() { return kernel_attr; }

bool CdistCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  in_data0_ = inputs[0]->addr;
  in_data1_ = inputs[1]->addr;
  out_data_ = outputs[0]->addr;
  auto task = [&](size_t start, size_t end) { LaunchKernel(start, end); };
  ParallelLaunchAutoSearch(task, static_cast<size_t>(batch_), this, &parallel_search_info_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cdist, CdistCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
