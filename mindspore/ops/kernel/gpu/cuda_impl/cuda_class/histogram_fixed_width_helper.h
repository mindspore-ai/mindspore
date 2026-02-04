/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HISTOGRAM_FIXED_WIDTH_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HISTOGRAM_FIXED_WIDTH_HELPER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include "kernel/gpu/cuda_impl/cuda_class/helper_base.h"
#include "kernel/gpu/cuda_impl/cuda_ops/histogram_fixed_width_impl.cuh"

#define INPUT_RANGE_SIZE 2

namespace mindspore {
namespace cukernel {
class HistogramFixedWidthAttr : public GpuKernelAttrBase {
 public:
  HistogramFixedWidthAttr() = default;
  ~HistogramFixedWidthAttr() override = default;
  int32_t nbins;
};

template <typename T>
class HistogramFixedWidthHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit HistogramFixedWidthHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~HistogramFixedWidthHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int out_flag =
      CalShapesSizeInBytes<int32_t>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (HasZeroInShapes(input_shapes) || out_flag == 1);
    input_x_shape_ = input_shapes[0];
    input_range_shape_ = input_shapes[1];
    if (input_range_shape_.size() != 1 || input_range_shape_[0] != INPUT_RANGE_SIZE) {
      MS_LOG(ERROR) << "For HistogramFixedWidth, shape of range mast be [2],but get" << input_range_shape_;
      return -1;
    }
    size_t work_size = level_size_ * sizeof(double);
    work_size_list_.emplace_back(work_size);

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_x_ptr = nullptr;
    T *input_range_ptr = nullptr;
    int32_t *output_ptr = nullptr;
    double *levels_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_x_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &input_range_ptr);
    if (flag != 0) {
      return flag;
    }

    std::unique_ptr<T[]> h_range_ptr = std::make_unique<T[]>(INPUT_RANGE_SIZE);
    cudaMemcpyAsync(h_range_ptr.get(), input_range_ptr, INPUT_RANGE_SIZE * sizeof(T), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream));
    if (h_range_ptr[0] >= h_range_ptr[1]) {
      MS_LOG(ERROR) << "For HistogramFixedWidth, range must satisfy range[0] < range[1], but get range[0] > range[1].";
      return -1;
    }

    flag = GetDeviceAddress<int32_t>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<double>(work_ptrs, 0, kernel_name_, &levels_ptr);
    if (flag != 0) {
      return flag;
    }

    d_levels_ = std::vector<double>(level_size_);
    const double step = static_cast<double>(h_range_ptr[1] - h_range_ptr[0]) / static_cast<double>(nbins_);
    d_levels_[0] = std::numeric_limits<double>::lowest();
    for (int i = 1; i < nbins_; i++) {
      d_levels_[i] = static_cast<double>(h_range_ptr[0]) + step * i;
    }
    d_levels_[nbins_] = std::numeric_limits<double>::max();
    cudaMemcpyAsync(levels_ptr, &d_levels_[0], level_size_ * sizeof(double), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream));
    int inputx_num_ = 1;
    for (int64_t i = input_x_shape_.size() - 1; i >= 0; i--) {
      inputx_num_ *= input_x_shape_[i];
    }

    // call cuda kernel
    auto status = CalHistogramFixedWidth(inputx_num_, input_x_ptr, levels_ptr, output_ptr, level_size_,
                                         reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<HistogramFixedWidthAttr>(kernel_attr);
    nbins_ = attr_ptr_->nbins;
    level_size_ = nbins_ + 1;
  }

  void ResetResource() {
    output_size_list_.clear();
    work_size_list_.clear();
    input_x_shape_.clear();
    input_range_shape_.clear();
  }

 private:
  std::shared_ptr<HistogramFixedWidthAttr> attr_ptr_;
  int32_t nbins_;
  std::vector<double> d_levels_;
  int64_t level_size_;
  bool is_null_input_;
  std::vector<int64_t> input_x_shape_;
  std::vector<int64_t> input_range_shape_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HISTOGRAM_FIXED_WIDTH_HELPER_H_
