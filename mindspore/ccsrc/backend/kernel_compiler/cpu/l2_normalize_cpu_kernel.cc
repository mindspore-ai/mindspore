/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/l2_normalize_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void L2NormalizeCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  epsilon_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  axis_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis"));
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  CheckParam(kernel_node);
  if (axis_ < 0) {
    axis_ += SizeToInt(input_shape_.size());
  }
}

template <typename T>
void L2NormalizeCPUKernel<T>::CalcDenominator(const T *input_addr, const size_t reduce_size, const int dims,
                                              std::unique_ptr<T[]> *denominator_addr) {
  T temp = (T)0.0;
  T epsilon = (T)epsilon_;
  T denominator = (T)0.0;
  // Calculate transpose axes and stride
  size_t stride = 1;
  std::vector<size_t> axes(input_shape_.size());
  int k = 0;
  for (int i = 0; i < dims; ++i) {
    if (i != axis_) {
      axes[k] = i;
      ++k;
    } else {
      stride *= input_shape_[i];
    }
  }
  axes[k] = axis_;

  std::vector<size_t> transpose_shape(input_shape_.size());
  for (int i = 0; i < dims; ++i) {
    transpose_shape[i] = input_shape_[axes[i]];
  }

  TransposeIterator tran_base_iter(std::move(transpose_shape), std::move(axes), input_shape_);

  auto task = [&](size_t start, size_t end) {
    auto iter = tran_base_iter;
    iter.SetPos(start * stride);
    for (size_t i = start; i < end; ++i) {
      denominator = input_addr[iter.GetPos()];
      denominator = denominator * denominator;
      iter.GenNextPos();
      for (size_t j = 1; j < stride; ++j) {
        temp = input_addr[iter.GetPos()];
        denominator += temp * temp;
        iter.GenNextPos();
      }
      denominator = (denominator > epsilon) ? denominator : epsilon;
      (*denominator_addr)[i] = sqrt(denominator);
    }
  };
  CPUKernelUtils::ParallelFor(task, reduce_size);
}

template <typename T>
void L2NormalizeCPUKernel<T>::CalcOutput(const T *input_addr, const std::vector<size_t> reduce_shape,
                                         const size_t output_size, T *output_addr,
                                         std::unique_ptr<T[]> const &denominator_addr) {
  BroadcastIterator broad_base_iter(input_shape_, reduce_shape, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = broad_base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      T dividend = input_addr[iter.GetInputPosA()];
      T divisor = denominator_addr[iter.GetInputPosB()];
      if (divisor == (T)0) {
        if (dividend == (T)0) {
          output_addr[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          output_addr[i] = dividend > (T)0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          output_addr[i] = dividend > (T)0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      output_addr[i] = dividend / divisor;
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size);
}

template <typename T>
bool L2NormalizeCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /*workspace*/,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  int dims = input_shape_.size();
  std::vector<size_t> reduce_shape = input_shape_;
  size_t reduce_size = 1;
  reduce_shape[axis_] = 1;
  for (int i = 0; i < dims; ++i) {
    reduce_size *= reduce_shape[i];
  }
  auto denominator_addr = std::make_unique<T[]>(reduce_size);

  L2NormalizeCPUKernel<T>::CalcDenominator(input_addr, reduce_size, dims, &denominator_addr);

  size_t output_size = outputs[0]->size / sizeof(T);
  L2NormalizeCPUKernel<T>::CalcOutput(input_addr, reduce_shape, output_size, output_addr, denominator_addr);

  return true;
}

template <typename T>
void L2NormalizeCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  int dims = SizeToInt(input_shape_.size());
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but L2NormalizeCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but L2NormalizeCPUKernel needs 1 output.";
  }
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "Attr axis_ " << axis_ << " must be in " << -dims << "~" << dims;
  }
  if (epsilon_ == 0.0) {
    MS_LOG(EXCEPTION) << "Attr epsilon can not be zero.";
  }
}
}  // namespace kernel
}  // namespace mindspore
