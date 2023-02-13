/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/maximum_grad_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaximumGradInputsNum = 3;
constexpr size_t kMaximumGradOutputsNum = 2;

void CheckShape(ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->empty()) {
    shape->push_back(1);
  }
}
}  // namespace

bool MaximumGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kMaximumGradInputsNum || outputs.size() != kMaximumGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaximumGradInputsNum << " and "
                  << kMaximumGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

int MaximumGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[kIndex0]->GetShapeVector();
  y_shape_ = inputs[kIndex1]->GetShapeVector();
  dout_shape = inputs[kIndex2]->GetShapeVector();
  dx_shape = outputs[kIndex0]->GetShapeVector();
  dy_shape = outputs[kIndex1]->GetShapeVector();
  CheckShape(&x_shape_);
  CheckShape(&y_shape_);
  CheckShape(&dout_shape);
  return KRET_OK;
}

bool MaximumGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaximumGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaximumGradOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    LaunchKernel<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    LaunchKernel<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    LaunchKernel<uint16_t>(inputs, outputs);
  }
  return true;
}

template <typename T>
void MaximumGradRecTask(const T *x, const T *y, const T *dout, T *dx, T *dy, size_t dim, size_t x_index, size_t y_index,
                        size_t dout_index, const std::vector<size_t> &x_cargo, const std::vector<size_t> &y_cargo,
                        const std::vector<size_t> &dout_cargo, const std::vector<size_t> &x_shape,
                        const std::vector<size_t> &y_shape, const std::vector<size_t> &dout_shape) {
  for (size_t i = 0; i < dout_shape[dim]; i++) {
    size_t x_i = x_shape[dim] == dout_shape[dim] ? i * x_cargo[dim] : 0;
    size_t y_i = y_shape[dim] == dout_shape[dim] ? i * y_cargo[dim] : 0;
    size_t dout_i = i * dout_cargo[dim];

    if (dim == dout_shape.size() - 1) {
      if (*(x + x_index + x_i) > *(y + y_index + y_i)) {
        *(dx + x_index + x_i) += *(dout + dout_index + i);
      } else if (*(x + x_index + x_i) == *(y + y_index + y_i)) {
        *(dx + x_index + x_i) += *(dout + dout_index + i) / 2;
        *(dy + y_index + y_i) += *(dout + dout_index + i) / 2;
      } else {
        *(dy + y_index + y_i) += *(dout + dout_index + i);
      }
    } else {
      MaximumGradRecTask(x, y, dout, dx, dy, dim + 1, x_index + x_i, y_index + y_i, dout_index + dout_i, x_cargo,
                         y_cargo, dout_cargo, x_shape, y_shape, dout_shape);
    }
  }
}

void GetCargo(std::vector<size_t> *const cargo, const std::vector<size_t> &shape,
              const std::vector<size_t> &dout_shape) {
  int i = dout_shape.size() - 1;
  int j = shape.size() - 1;
  (*cargo)[i] = 1;
  for (--i; j >= 1; --i, --j) {
    (*cargo)[i] = shape[j] * (*cargo)[i + 1];
  }
  for (; i >= 0; i--) {
    (*cargo)[i] = 1;
  }
}

size_t GetTensorLen(const ShapeVector &shape) {
  int64_t len = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    len *= shape[i];
  }
  return LongToSize(len);
}

void GetShape(std::vector<size_t> *shape, const ShapeVector &shape_, const ShapeVector &dout_shape) {
  int k = dout_shape.size() - 1;
  int i = shape_.size() - 1;
  for (; i >= 0; i--, k--) {
    (*shape)[IntToSize(k)] = LongToSize(shape_[IntToSize(i)]);
  }
}

template <typename T>
void MaximumGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto dout_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto dx_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto dy_addr = reinterpret_cast<T *>(outputs[1]->addr);

  size_t x_tensor_len = GetTensorLen(x_shape_);
  size_t y_tensor_len = GetTensorLen(y_shape_);
  size_t x_tensor_size = x_tensor_len * sizeof(T);
  size_t y_tensor_size = y_tensor_len * sizeof(T);
  auto res_dx = memset_s(dx_addr, x_tensor_size, 0, x_tensor_size);
  auto res_dy = memset_s(dy_addr, y_tensor_size, 0, y_tensor_size);
  if (res_dx != EOK || res_dy != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no of x: " << res_dx
                      << " and y: " << res_dy;
  }
  std::vector<size_t> x_shape(dout_shape.size(), 1);
  std::vector<size_t> y_shape(dout_shape.size(), 1);
  std::vector<size_t> x_cargo(dout_shape.size(), 0);
  std::vector<size_t> y_cargo(dout_shape.size(), 0);
  std::vector<size_t> dout_cargo(dout_shape.size(), 0);
  auto dout_shape_sizet = Convert2SizeT(dout_shape);

  GetShape(&x_shape, x_shape_, dout_shape);
  GetShape(&y_shape, y_shape_, dout_shape);

  GetCargo(&x_cargo, x_shape, dout_shape_sizet);
  GetCargo(&y_cargo, y_shape, dout_shape_sizet);
  GetCargo(&dout_cargo, dout_shape_sizet, dout_shape_sizet);

  MaximumGradRecTask<T>(x_addr, y_addr, dout_addr, dx_addr, dy_addr, 0, 0, 0, 0, x_cargo, y_cargo, dout_cargo, x_shape,
                        y_shape, dout_shape_sizet);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaximumGrad, MaximumGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
