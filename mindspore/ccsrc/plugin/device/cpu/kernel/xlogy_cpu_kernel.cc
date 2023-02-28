/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/xlogy_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Eigen"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
static constexpr int MAX_DIMS = 7;
static constexpr size_t PARALLEL_THRESHOLD = 4096;

template <typename T>
void XlogySameShapeTask(T *x_addr, T *y_addr, T *output_addr, size_t start, size_t end) {
  Eigen::Map<Eigen::Array<T, -1, 1>> x_v(x_addr + start, end - start);
  Eigen::Map<Eigen::Array<T, -1, 1>> y_v(y_addr + start, end - start);
  Eigen::Map<Eigen::Array<T, -1, 1>> o_v(output_addr + start, end - start);
  o_v = x_v * y_v.log();
}

template <>
void XlogySameShapeTask(float16 *x_addr, float16 *y_addr, float16 *output_addr, size_t start, size_t end) {
  Eigen::half *ex_addr = reinterpret_cast<Eigen::half *>(x_addr);
  Eigen::half *ey_addr = reinterpret_cast<Eigen::half *>(y_addr);
  Eigen::half *eo_addr = reinterpret_cast<Eigen::half *>(output_addr);
  Eigen::Map<Eigen::Array<Eigen::half, -1, 1>> x_v(ex_addr + start, end - start);
  Eigen::Map<Eigen::Array<Eigen::half, -1, 1>> y_v(ey_addr + start, end - start);
  Eigen::Map<Eigen::Array<Eigen::half, -1, 1>> o_v(eo_addr + start, end - start);
  o_v = x_v * y_v.log();
}

template <typename T>
bool XlogyCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  if (has_null_input_) {
    return true;
  }
  auto x_addr = static_cast<T *>(inputs[0]->addr);
  auto y_addr = static_cast<T *>(inputs[1]->addr);
  auto output_addr = static_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  auto sameShapeTask = [&x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    XlogySameShapeTask(x_addr, y_addr, output_addr, start, end);
  };
  auto diffShapeTask = [this, &x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto x1 = x_addr[index_listx_[i]];
      auto x2 = y_addr[index_listy_[i]];
      auto logx2 = log(x2);
      output_addr[i] = x1 * logx2;
    }
  };

  CTask task = is_need_broadcast_ ? CTask(diffShapeTask) : CTask(sameShapeTask);
  if (output_size < PARALLEL_THRESHOLD) {
    task(0, output_size);
  } else {
    ParallelLaunch(task, output_size, PARALLEL_THRESHOLD, this, pool_);
  }
  return true;
}

bool XlogyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

bool XlogyCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_type = inputs[0]->GetDtype();
  auto y_type = inputs[1]->GetDtype();
  auto out_type = outputs[0]->GetDtype();
  if (!(x_type == y_type && x_type == out_type)) {
    MS_LOG(ERROR) << "Xlogy need same input and output data type, but got X type:" << x_type << " Y type:" << y_type
                  << " out type:" << out_type;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int XlogyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto y_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());

  // while has null input, xlogy result is null too
  has_null_input_ = CheckNullInput(x_shape);
  has_null_input_ = has_null_input_ || CheckNullInput(y_shape);
  if (has_null_input_) {
    return 0;
  }

  auto out_shape = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  if (out_shape.size() > MAX_DIMS || out_shape.size() < x_shape.size() || out_shape.size() < y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", and output dimension can't less than input; but got x_shape dimension:" << x_shape.size()
                      << " ,y_shape dimension:" << y_shape.size() << " ,out_shape dimension:" << out_shape.size();
  }
  is_need_broadcast_ = x_shape != y_shape;
  if (is_need_broadcast_) {
    GetBroadCastIndex(x_shape, out_shape, &index_listx_);
    GetBroadCastIndex(y_shape, out_shape, &index_listy_);
  }
  return 0;
}

const std::vector<std::pair<KernelAttr, XlogyCpuKernelMod::KernelRunFunc>> &XlogyCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, XlogyCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &XlogyCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &XlogyCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &XlogyCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &XlogyCpuKernelMod::LaunchKernel<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &XlogyCpuKernelMod::LaunchKernel<complex128>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Xlogy, XlogyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
