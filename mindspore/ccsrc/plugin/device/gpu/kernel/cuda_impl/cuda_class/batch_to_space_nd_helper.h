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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCH_TO_SPACE_ND_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCH_TO_SPACE_ND_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/batch_to_space_nd_impl.cuh"

namespace mindspore {
namespace cukernel {
class BatchToSpaceNDAttr : public GpuKernelAttrBase {
 public:
  BatchToSpaceNDAttr() = default;
  ~BatchToSpaceNDAttr() override = default;
  std::vector<std::vector<int64_t>> crops;
  std::vector<int64_t> block_shape;
};

template <typename T>
class BatchToSpaceNDHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit BatchToSpaceNDHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~BatchToSpaceNDHelperGpuKernel() = default;
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
    for (size_t i = 0; i < input_shape_.size(); ++i) {
      input_size_ = input_shape_[i] * input_size_;
    }

    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
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

    std::vector<int64_t> crops_start_(crops_.size(), 0);
    for (int i = 0; i < static_cast<int>(crops_.size()); i++) crops_start_[i] = crops_[i][0];
    std::vector<int64_t> stride_(input_shape_size, 1);
    for (int i = static_cast<int>(input_shape_size) - 2; i >= 0; i--) stride_[i] = stride_[i + 1] * input_shape_[i + 1];
    std::vector<int64_t> on_stride_(block_rank_, 1);
    if (block_rank_ > 1) {
      for (int i = static_cast<int>(block_rank_) - 2; i >= 0; i--)
        on_stride_[i] = on_stride_[i + 1] * block_shape_[i + 1];
    }
    // call cuda kernel
    auto status = CalBatchToSpaceND(input_ptr, crops_start_.data(), block_shape_.data(), output_shape_.data(),
                                    output_shape_size, stride_.data(), on_stride_.data(), off_set_, output_size_,
                                    output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<BatchToSpaceNDAttr>(kernel_attr);
  }

  void ResetResource() noexcept override {
    block_rank_ = 0;
    off_set_ = 0;
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
    constexpr size_t CROP_SHAPE_1 = 2;
    crops_ = attr_ptr_->crops;
    block_shape_ = attr_ptr_->block_shape;
    block_rank_ = block_shape_.size();
    off_set_ = input_shape_.size() - block_shape_.size();

    if (static_cast<int>(block_shape_.size()) - static_cast<int>(input_shape_.size()) >= 0) {
      MS_LOG(ERROR) << kernel_name_ << " resize failed because input shape should be greater than block shape, "
                    << "but input shape is " << input_shape_ << " and block shape is " << block_shape_;
      return -1;
    }

    // check crops_
    if (crops_.size() != block_rank_) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the size of 'crops' should be equal to the length of 'block_shape':  " << block_rank_
                    << ", but got " << crops_.size();
      return -1;
    }
    int64_t block_shape_prod = 1;
    for (size_t idx_i = 0; idx_i < block_rank_; ++idx_i) {
      if (block_shape_[idx_i] < 1) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', the elements of 'block_shape' should be both larger than 1, but got " << idx_i
                      << "'th block size " << block_shape_[idx_i] << ")\n";
        return -1;
      }
      block_shape_prod = block_shape_prod * block_shape_[idx_i];
      if (crops_[idx_i].size() != CROP_SHAPE_1) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', the size of each vector of 'crops' should be equal to the length of 'block_shape': "
                      << CROP_SHAPE_1 << ", but got " << idx_i << "'th element: " << crops_[idx_i].size();
        return -1;
      }
      for (size_t idx_j = 0; idx_j < CROP_SHAPE_1; ++idx_j) {
        if (crops_[idx_i][idx_j] < 0) {
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', the element of 'crops' cannot be less than 0, "
                        << "but got crops[" << idx_i << "][ " << idx_j << "]: " << crops_[idx_i][idx_j];
          return -1;
        }
      }
    }

    if (input_shape_[0] % block_shape_prod != 0) {
      MS_LOG(ERROR)
        << "For '" << kernel_name_
        << "', the first dim of 'input_x' must be divisible by 'block_shape_prod'. But got first dim of 'input_x': "
        << input_shape_[0] << ", 'block_shape_prod' with value: " << block_shape_prod << ".";
      return -1;
    }
    return 0;
  }

 private:
  std::shared_ptr<BatchToSpaceNDAttr> attr_ptr_;
  std::vector<std::vector<int64_t>> crops_;
  std::vector<int64_t> block_shape_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  size_t block_rank_;
  size_t off_set_;
  size_t input_shape_size;
  size_t output_shape_size;
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCH_TO_SPACE_ND_HELPER_H_
