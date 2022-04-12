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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <functional>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/index_fill_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr int kIndexFillInputsNum = 4;
constexpr int kIndexFillOutputsNum = 1;
constexpr int kIndexFillX = 0;      // x
constexpr int kIndexFillDim = 1;    // dim
constexpr int kIndexFillIndex = 2;  // index
constexpr int kIndexFillValue = 3;  // value
constexpr int kIndexFillY = 0;      // output
using kIndexType = int32_t;
template <typename T>
class IndexFillGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  IndexFillGpuKernelMod() { ResetResource(); }
  ~IndexFillGpuKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    // 'dim' and 'value' must be tensor with empty shape.
    auto dim_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexFillDim);
    if (!(dim_shape.empty() || (dim_shape.size() == 1 && dim_shape.front() == 1))) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'dim' should be empty or (1,), but got "
                        << CONVERT_VECTOR_TO_STRING(dim_shape);
    }
    auto value_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexFillValue);
    if (!(value_shape.empty() || (value_shape.size() == 1 && value_shape.front() == 1))) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'value' should be empty or (1,), but got "
                        << CONVERT_VECTOR_TO_STRING(value_shape);
    }
    // 'index' must be vector/scalar.
    auto index_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexFillIndex);
    if (index_shape.size() > 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'index' should not be greater than 1D, but got "
                        << CONVERT_VECTOR_TO_STRING(index_shape);
    }
    // The shape of 'x' and 'y' is equal.
    auto x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexFillX);
    auto y_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kIndexFillY);
    bool is_equal = x_shape.size() == y_shape.size();
    for (size_t i = 0; is_equal && i < x_shape.size(); ++i) {
      is_equal &= (x_shape[i] == y_shape[i]);
    }
    if (!is_equal) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'x' and 'y' must be equal, but got "
                        << "shape of 'x' is " << CONVERT_VECTOR_TO_STRING(x_shape) << ", shape of 'output' is "
                        << CONVERT_VECTOR_TO_STRING(y_shape);
    }

    is_null_input_ = CHECK_SHAPE_NULL(x_shape_, kernel_name_, "x");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    x_size_ = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<size_t>());
    index_size_ = std::accumulate(index_shape.begin(), index_shape.end(), 1, std::multiplies<size_t>());
    x_shape_ = x_shape;
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexFillInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexFillOutputsNum, kernel_name_);
    if (is_null_input_) {
      return true;
    }
    auto x_ptr = GetDeviceAddress<T>(inputs, kIndexFillX);
    auto dim_ptr = GetDeviceAddress<kIndexType>(inputs, kIndexFillDim);
    auto index_ptr = GetDeviceAddress<kIndexType>(inputs, kIndexFillIndex);
    auto value_ptr = GetDeviceAddress<T>(inputs, kIndexFillValue);
    auto y_ptr = GetDeviceAddress<T>(outputs, kIndexFillY);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Check and Initialize 'dim'.
    kIndexType dim, rank = SizeToInt(x_shape_.size());
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&dim, dim_ptr, sizeof(kIndexType), cudaMemcpyDeviceToHost, cuda_stream),
                               "cudaMemcpyAsync input 'dim' to host failed");
    if (dim < -rank || dim >= rank) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'dim' should be in the range [-" << rank << "," << rank
                        << "), but got " << dim;
    }

    // Initialize 'value'.
    T value;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&value, value_ptr, sizeof(T), cudaMemcpyDeviceToHost, cuda_stream),
                               "cudaMemcpyAsync input 'value' to host failed");

    // Copy from 'x' into 'y'.
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(y_ptr, x_ptr, x_size_ * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream),
      "cudaMemcpyAsync output 'y' from 'x' failed.");

    // Prepare index_num, dim_size, outer_size, inner_size
    dim = dim >= 0 ? dim : dim + rank;
    int dim_size = 1;
    size_t outer_size = 1, inner_size = 1;
    for (int i = 0; i < rank; i++) {
      if (i < dim) {
        outer_size *= x_shape_.at(IntToSize(i));
      } else if (i > dim) {
        inner_size *= x_shape_.at(IntToSize(i));
      } else {
        dim_size = static_cast<kIndexType>(x_shape_.at(IntToSize(i)));
      }
    }
    size_t index_num = index_size_ * (outer_size * inner_size);

    bool out_bound = IndexFill(y_ptr, index_ptr, index_num, outer_size, dim_size, inner_size, value, cuda_stream);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaGetLastError(), "Implementing IndexFill failed");
    if (out_bound) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input 'index' out of bounds";
    }
    return true;
  }

  void ResetResource() noexcept override {
    is_null_input_ = false;
    x_size_ = 0;
    index_size_ = 0;
    x_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_ * sizeof(T));
    input_size_list_.push_back(sizeof(kIndexType));
    input_size_list_.push_back(index_size_ * sizeof(kIndexType));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(x_size_ * sizeof(T));
  }

 private:
  size_t x_size_;
  size_t index_size_;
  std::vector<size_t> x_shape_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_INDEX_FILL_GPU_KERNEL_H_
