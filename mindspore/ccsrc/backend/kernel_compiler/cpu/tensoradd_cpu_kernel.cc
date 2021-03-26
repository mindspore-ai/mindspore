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
#include "backend/kernel_compiler/cpu/tensoradd_cpu_kernel.h"
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
struct Iterator {
  std::vector<size_t> coordinates_;
  std::vector<size_t> input_shape_a_;
  std::vector<size_t> input_shape_b_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> input_strides_a_;
  std::vector<size_t> input_strides_b_;
  int output_dimension_pos_{0};
  size_t pos_{0};
  Iterator(const std::vector<size_t> &input_shape_a, const std::vector<size_t> &input_shape_b,
           const std::vector<size_t> &output_shape, const std::vector<size_t> &input_strides_a,
           const std::vector<size_t> &input_strides_b, size_t pos)
      : input_shape_a_(input_shape_a),
        input_shape_b_(input_shape_b),
        output_shape_(output_shape),
        input_strides_a_(input_strides_a),
        input_strides_b_(input_strides_b),
        pos_{pos} {
    output_dimension_pos_ = output_shape.size() - 1;
    // Calculate coordinate with pos
    coordinates_.resize(output_dimension_pos_ + 1);
    int tmp = pos_;
    for (int i = output_dimension_pos_; i >= 0 && tmp != 0; --i) {
      coordinates_[i] = tmp % output_shape_[i];
      tmp /= output_shape_[i];
    }
  }

  void UpdateCoordinates() {
    // Calculate output next coordinate
    for (int i = output_dimension_pos_; i >= 0; --i) {
      if (coordinates_[i] + 1 == output_shape_[i]) {
        coordinates_[i] = 0;
      } else {
        ++coordinates_[i];
        break;
      }
    }
  }

  void GenPoints(std::array<size_t, 2> *position) {
    auto &idx = *position;
    idx = {0, 0};
    for (int k = 0; k < output_dimension_pos_; ++k) {
      if (input_shape_a_[k] > 1) {
        idx[0] += coordinates_[k] * input_strides_a_[k];
      }
      if (input_shape_b_[k] > 1) {
        idx[1] += coordinates_[k] * input_strides_b_[k];
      }
    }
    if (input_shape_a_[output_dimension_pos_] > 1) {
      idx[0] += coordinates_[output_dimension_pos_];
    }
    if (input_shape_b_[output_dimension_pos_] > 1) {
      idx[1] += coordinates_[output_dimension_pos_];
    }
  }
};
}  // namespace

void TensorAddCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Init shape ans strides
  input_shape_a_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_shape_b_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
}

bool TensorAddCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr_a = reinterpret_cast<float *>(inputs[0]->addr);
  auto input_addr_b = reinterpret_cast<float *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  auto output_size = outputs[0]->size / sizeof(float);
  if (input_shape_a_ == input_shape_b_) {
    NormalProcess(input_addr_a, input_addr_b, output_addr, output_size);
  } else {  // Broadcast
    BroadcastProcess(input_addr_a, input_addr_b, output_addr, output_size);
  }
  return true;
}

void TensorAddCPUKernel::NormalProcess(const float *input_a, const float *input_b, float *output, size_t size) {
  auto task = [output, input_a, input_b](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      output[i] = input_a[i] + input_b[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

void TensorAddCPUKernel::BroadcastProcess(const float *input_a, const float *input_b, float *output, size_t size) {
  // Broadcast shape
  int dimension = output_shape_.size();
  int input_dimension_a = input_shape_a_.size();
  if (input_dimension_a < dimension) {
    input_shape_a_.insert(input_shape_a_.begin(), dimension - input_dimension_a, 1);
  }
  int input_dimension_b = input_shape_b_.size();
  if (input_dimension_b < dimension) {
    input_shape_b_.insert(input_shape_b_.begin(), dimension - input_dimension_b, 1);
  }

  // Calculate strides
  CalculateStrides(input_shape_a_, &input_strides_a_);
  CalculateStrides(input_shape_b_, &input_strides_b_);

  auto task = [this, input_a, input_b, output](size_t start, size_t end) {
    Iterator iter(input_shape_a_, input_shape_b_, output_shape_, input_strides_a_, input_strides_b_, start);
    std::array<size_t, 2> position{0};
    for (size_t i = start; i < end; ++i) {
      iter.GenPoints(&position);
      output[i] = input_a[position[0]] + input_b[position[1]];
      iter.UpdateCoordinates();
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

void TensorAddCPUKernel::CalculateStrides(const std::vector<size_t> &shape, std::vector<size_t> *strides) {
  strides->resize(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    (*strides)[i] = shape[i + 1] * (*strides)[i + 1];
  }
}
}  // namespace kernel
}  // namespace mindspore
