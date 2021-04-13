/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/minimum_grad_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
void GetCargo(std::vector<size_t> *cargo, const std::vector<size_t> &shape, const std::vector<size_t> &dout_shape) {
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

size_t GetTensorLen(const std::vector<size_t> &shape) {
  size_t len = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    len *= shape[i];
  }
  return len;
}

void GetShape(std::vector<size_t> *shape, const std::vector<size_t> &shape_, const std::vector<size_t> &dout_shape) {
  int k = dout_shape.size() - 1;
  int i = shape_.size() - 1;
  for (; i >= 0; i--, k--) {
    (*shape)[k] = shape_[i];
  }
}
}  // namespace

void MinimumGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  y_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  dout_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  dx_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dy_shape = AnfAlgo::GetOutputInferShape(kernel_node, 1);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (!x_shape_.size() || !y_shape_.size() || !dout_shape.size()) {
    MS_LOG(EXCEPTION) << "Input NULL";
  }
}

bool MinimumGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
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
  }
  return true;
}

template <typename T>
void MinimumGradRecTask(const T *x, const T *y, const T *dout, T *dx, T *dy, const size_t dim, const size_t x_index,
                        const size_t y_index, const size_t dout_index, const std::vector<size_t> &x_cargo,
                        const std::vector<size_t> &y_cargo, const std::vector<size_t> &dout_cargo,
                        const std::vector<size_t> &x_shape, const std::vector<size_t> &y_shape,
                        const std::vector<size_t> &dout_shape) {
  for (size_t i = 0; i < dout_shape[dim]; i++) {
    size_t x_i = x_shape[dim] == dout_shape[dim] ? i * x_cargo[dim] : 0;
    size_t y_i = y_shape[dim] == dout_shape[dim] ? i * y_cargo[dim] : 0;
    size_t dout_i = i * dout_cargo[dim];

    if (dim == dout_shape.size() - 1) {
      if (*(x + x_index + x_i) <= *(y + y_index + y_i)) {
        *(dx + x_index + x_i) += *(dout + dout_index + i);
      } else {
        *(dy + y_index + y_i) += *(dout + dout_index + i);
      }
    } else {
      MinimumGradRecTask(x, y, dout, dx, dy, dim + 1, x_index + x_i, y_index + y_i, dout_index + dout_i, x_cargo,
                         y_cargo, dout_cargo, x_shape, y_shape, dout_shape);
    }
  }
}

template <typename T>
void MinimumGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto dout_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto dx_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto dy_addr = reinterpret_cast<T *>(outputs[1]->addr);

  size_t x_tensor_len = GetTensorLen(x_shape_);
  size_t y_tensor_len = GetTensorLen(y_shape_);
  memset_s(dx_addr, x_tensor_len * sizeof(T), 0x00, x_tensor_len * sizeof(T));
  memset_s(dy_addr, y_tensor_len * sizeof(T), 0x00, y_tensor_len * sizeof(T));

  std::vector<size_t> x_shape(dout_shape.size(), 1);
  std::vector<size_t> y_shape(dout_shape.size(), 1);
  std::vector<size_t> x_cargo(dout_shape.size(), 0);
  std::vector<size_t> y_cargo(dout_shape.size(), 0);
  std::vector<size_t> dout_cargo(dout_shape.size(), 0);

  GetShape(&x_shape, x_shape_, dout_shape);
  GetShape(&y_shape, y_shape_, dout_shape);

  GetCargo(&x_cargo, x_shape, dout_shape);
  GetCargo(&y_cargo, y_shape, dout_shape);
  GetCargo(&dout_cargo, dout_shape, dout_shape);

  MinimumGradRecTask<T>(x_addr, y_addr, dout_addr, dx_addr, dy_addr, 0, 0, 0, 0, x_cargo, y_cargo, dout_cargo, x_shape,
                        y_shape, dout_shape);
}

void MinimumGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but MinimumGradCPUKernel needs 3 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but MinimumGradCPUKernel needs 2 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
