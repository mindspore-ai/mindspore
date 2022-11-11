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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPACE_TO_BATCH_ND_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPACE_TO_BATCH_ND_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/space_to_batch_nd_impl.cuh"

namespace mindspore {
namespace cukernel {
class SpaceToBatchNDAttr : public GpuKernelAttrBase {
 public:
  SpaceToBatchNDAttr() = default;
  ~SpaceToBatchNDAttr() override = default;
  std::vector<std::vector<int64_t>> paddings;
  std::vector<int64_t> block_shape;
};

template <typename T>
class SpaceToBatchNDHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SpaceToBatchNDHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~SpaceToBatchNDHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    input_shape_ = input_shapes[0];

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    output_shape_ = output_shapes[0];
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    input_size_ = 1;
    for (size_t i = 0; i < (size_t) static_cast<int64_t>(input_shape_.size()); ++i) {
      input_size_ = input_shape_[i] * input_size_;
    }

    output_size_ = 1;
    for (size_t i = 0; i < (size_t) static_cast<int64_t>(output_shape_.size()); ++i) {
      output_size_ = output_shape_[i] * output_size_;
    }

    T *input_ptr = nullptr;
    T *output_ptr = nullptr;
    input_shape_size = input_shape_.size();
    output_shape_size = output_shape_.size();

    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    std::vector<int64_t> paddings_start(paddings_.size(), 0);
    for (int i = 0; i < static_cast<int>(paddings_.size()); i++) paddings_start[i] = paddings_[i][0];
    std::vector<int64_t> stride(output_shape_size, 1);
    for (int i = static_cast<int>(output_shape_size) - 2; i >= 0; i--) stride[i] = stride[i + 1] * output_shape_[i + 1];
    std::vector<int64_t> on_stride(block_rank, 1);
    if (block_rank > 1) {
      for (int i = static_cast<int>(block_rank) - 2; i >= 0; i--) on_stride[i] = on_stride[i + 1] * block_shape_[i + 1];
    }

    // call cuda kernel
    CalSpaceToBatchND(input_ptr, paddings_start.data(), block_shape_.data(), input_shape_.data(), input_shape_size,
                      stride.data(), on_stride.data(), off_set, input_size_, output_size_, output_ptr, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<SpaceToBatchNDAttr>(kernel_attr);
  }

  void ResetResource() noexcept override {
    block_rank = 0;
    off_set = 0;
    input_size_ = 0;
    output_size_ = 0;
    input_shape_size = 0;
    output_shape_size = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 protected:
  int CheckKernelParam() override {
    constexpr size_t PADDING_SHAPE_1 = 2;
    paddings_ = attr_ptr_->paddings;
    block_shape_ = attr_ptr_->block_shape;
    block_rank = block_shape_.size();
    off_set = input_shape_.size() - block_shape_.size();

    // check paddings_
    if (paddings_.size() != block_rank) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the size of 'paddings' should be equal to the length of 'block_size':  " << block_rank
                    << ", but got " << paddings_.size();
      return -1;
    }

    for (size_t idx_i = 0; idx_i < block_rank; ++idx_i) {
      if (paddings_[idx_i].size() != PADDING_SHAPE_1) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', the size of each vector of 'paddings' should be equal to the length of 'block_size': "
                      << PADDING_SHAPE_1 << ", but got " << idx_i << "'th element: " << paddings_[idx_i].size();
        return -1;
      }
      for (size_t idx_j = 0; idx_j < PADDING_SHAPE_1; ++idx_j) {
        if (paddings_[idx_i][idx_j] < 0) {
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', the element of 'paddings' cannot be less than 0, "
                        << "but got paddings[" << idx_i << "][ " << idx_j << "]: " << paddings_[idx_i][idx_j];
          return -1;
        }
      }
      auto tmp_shape = input_shape_[idx_i + off_set] + paddings_[idx_i][0] + paddings_[idx_i][1];
      if ((tmp_shape % block_shape_[idx_i]) != 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', padded shape should be divisible by block_size, , but got padded shape: " << tmp_shape
                      << ", block_size: " << block_shape_[idx_i];
        return -1;
      }
      if ((tmp_shape / block_shape_[idx_i]) == 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', padded shape cannot be less than block_size"
                      << ", but got padded shape: " << tmp_shape << ", block_size: " << block_shape_[idx_i];
        return -1;
      }
    }

    return 0;
  }

 private:
  std::shared_ptr<SpaceToBatchNDAttr> attr_ptr_;
  std::vector<std::vector<int64_t>> paddings_;
  std::vector<int64_t> block_shape_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  size_t block_rank;
  size_t off_set;
  size_t input_shape_size;
  size_t output_shape_size;
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPACE_TO_BATCH_ND_HELPER_H_
