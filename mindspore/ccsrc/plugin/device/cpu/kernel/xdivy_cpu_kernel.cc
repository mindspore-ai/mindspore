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
#include "plugin/device/cpu/kernel/xdivy_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Eigen"

namespace mindspore {
namespace kernel {
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
static constexpr int MAX_DIMS = 7;
static constexpr size_t PARALLEL_THRESHOLD = 4096;
template <typename T>
T GetDivZeroVal(const T &v) {
  auto zero = static_cast<T>(0.0);
  return v > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
}

template <>
complex128 GetDivZeroVal(const complex128 &) {
  return std::numeric_limits<complex128>::quiet_NaN();
}

template <>
complex64 GetDivZeroVal(const complex64 &) {
  return std::numeric_limits<complex64>::quiet_NaN();
}

template <class T>
bool isZero(const T &val) {
  return val == T(0.0f);
}

template <>
bool isZero(const float &val) {
  return std::fpclassify(val) == FP_ZERO;
}

template <>
bool isZero(const double &val) {
  return std::fpclassify(val) == FP_ZERO;
}

template <typename T>
void SameShapeTask(T *x_addr, T *y_addr, T *output_addr, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    auto dividend = x_addr[i];
    auto divisor = y_addr[i];
    if (isZero(divisor)) {
      if (isZero(dividend)) {
        output_addr[i] = static_cast<T>(0.0);
        continue;
      }
      output_addr[i] = GetDivZeroVal(dividend);
      continue;
    }
    output_addr[i] = dividend / divisor;
  }
}

template <>
void SameShapeTask(float *x_addr, float *y_addr, float *output_addr, size_t start, size_t end) {
  Eigen::Map<Eigen::ArrayXf> x_v(x_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXf> y_v(y_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXf> o_v(output_addr + start, end - start);
  o_v = (x_v == 0).select(o_v, x_v / y_v);
}

template <>
void SameShapeTask(double *x_addr, double *y_addr, double *output_addr, size_t start, size_t end) {
  Eigen::Map<Eigen::ArrayXd> x_v(x_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXd> y_v(y_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXd> o_v(output_addr + start, end - start);
  o_v = (x_v == 0).select(o_v, x_v / y_v);
}

template <typename T>
bool XdivyCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_addr = static_cast<T *>(inputs[0]->addr);
  auto y_addr = static_cast<T *>(inputs[1]->addr);
  auto output_addr = static_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  auto sameShapeTask = [&x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    SameShapeTask(x_addr, y_addr, output_addr, start, end);
  };
  auto diffShapeTask = [this, &x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto idxX = index_listx_[i];
      auto idxY = index_listy_[i];
      auto dividend = x_addr[idxX];
      auto divisor = y_addr[idxY];
      auto zero = (T)0;
      if (divisor == zero) {
        if (dividend == zero) {
          output_addr[i] = zero;
          continue;
        }
        output_addr[i] = GetDivZeroVal(dividend);
        continue;
      }
      output_addr[i] = dividend / divisor;
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

bool XdivyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  const size_t kInputsNum = 2;
  const size_t kOutputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

bool XdivyCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_type = inputs[0]->GetDtype();
  auto y_type = inputs[1]->GetDtype();
  auto out_type = outputs[0]->GetDtype();
  if (!(x_type == y_type && x_type == out_type)) {
    MS_LOG(ERROR) << "Xdivy need same input and output data type, but got X type:" << x_type << " Y type:" << y_type
                  << " out type:" << out_type;
    return false;
  }

  auto iter = func_map_.find(x_type);
  if (iter == func_map_.end()) {
    MS_LOG(ERROR) << "Xdivy only support tensor with data type float16, float32, "
                     "float64, Complex64, Complex128, but got typeid:"
                  << x_type;
    return false;
  }
  kernel_func_ = iter->second;
  return true;
}

void GetBroadCastIndex(const ShapeVector &unaligned_input_shape, const ShapeVector &output_shape,
                       std::vector<int64_t> *index_list) {
  // Given unaligned input shape and output shape, this function returns the mapping
  // from indices of output (logical) to corespondingly real input indices (physical).
  // The return will write to index_list, whose size is equal to total elements of output.
  constexpr int MaxDim = 10;
  int64_t logical_shape[MaxDim];
  int64_t physical_shape[MaxDim];
  int64_t size = 0, output_size = 1;
  // Align input shape to output shape by filling one into the outermost dimension.
  ShapeVector input_shape(output_shape.size());
  for (size_t i = 0, j = output_shape.size() - unaligned_input_shape.size(); i < output_shape.size(); i++) {
    input_shape[i] = i < j ? 1 : unaligned_input_shape[i - j];
  }
  // Get logical shape and physical shape of input. Moreover, we will merge the dimensions with same
  // (logical or physical) property.
  for (int i = SizeToInt(output_shape.size()) - 1; i >= 0;) {
    int64_t stride = 1;
    bool change = false, is_valid = false;
    while (i >= 0 && input_shape[i] == output_shape[i]) {
      stride *= output_shape[i];
      change = is_valid = true;
      --i;
    }
    if (change) {
      output_size *= stride;
      logical_shape[size] = physical_shape[size] = stride;
      size++;
    }
    change = false;
    stride = 1;
    while (i >= 0 && input_shape[i] == 1) {
      stride *= output_shape[i];
      change = is_valid = true;
      --i;
    }
    if (change) {
      output_size *= stride;
      logical_shape[size] = 1;
      physical_shape[size] = stride;
      size++;
    }
    if (!is_valid) {
      MS_LOG(EXCEPTION) << "Both shape are not able to broadcast, input shape is " << unaligned_input_shape
                        << " and output shape is " << output_shape;
    }
  }
  // Get the flatten input indices according to "logical_shape" and "physical_shape".
  int64_t offset = 1;
  int64_t stride = 1;
  index_list->resize(output_size);
  (*index_list)[0] = 0;  // First element is set to 0.
  for (int64_t i = 0; i < size; ++i) {
    int64_t increment = (logical_shape[i] == physical_shape[i] ? stride : 0);
    for (int64_t j = 0; j < (physical_shape[i] - 1) * offset; ++j) {
      (*index_list)[offset + j] = (*index_list)[j] + increment;
    }
    offset *= physical_shape[i];
    stride *= logical_shape[i];
  }
}

int XdivyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  ResetResource();
  int ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto x_shape = inputs[0]->GetShapeVector();
  auto y_shape = inputs[1]->GetShapeVector();
  auto out_shape = outputs[0]->GetShapeVector();
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

std::vector<KernelAttr> XdivyCpuKernelMod::GetOpSupport() { return support_ops_; }

std::vector<KernelAttr> XdivyCpuKernelMod::support_ops_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)},
};

std::map<mindspore::TypeId, XdivyCpuKernelMod::XdivyFunc> XdivyCpuKernelMod::func_map_ = {
  {kNumberTypeFloat16, &XdivyCpuKernelMod::LaunchKernel<float16>},
  {kNumberTypeFloat32, &XdivyCpuKernelMod::LaunchKernel<float>},
  {kNumberTypeFloat64, &XdivyCpuKernelMod::LaunchKernel<double>},
  {kNumberTypeComplex64, &XdivyCpuKernelMod::LaunchKernel<complex64>},
  {kNumberTypeComplex128, &XdivyCpuKernelMod::LaunchKernel<complex128>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Xdivy, XdivyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
