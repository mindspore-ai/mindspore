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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_ADD_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_ADD_GRAD_HELPER_H_
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_add_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t kSparseAddGradIndex0 = 0;
constexpr size_t kSparseAddGradIndex1 = 1;
constexpr size_t kSparseAddGradIndex2 = 2;
constexpr size_t kSparseAddGradIndex3 = 3;
constexpr size_t kSparseAddGradIndicesDim = 2;
template <typename T, typename S>
class SparseAddGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SparseAddGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    index_bytes_ = sizeof(T);
    value_bytes_ = sizeof(S);
  }

  virtual ~SparseAddGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    input_shapes_ = input_shapes;
    output_shapes_ = output_shapes;
    int check_flag = CheckKernelParam();
    if (check_flag == -1) {
      return check_flag;
    }

    (void)std::transform(input_shapes.at(kSparseAddGradIndex0).begin(), input_shapes.at(kSparseAddGradIndex0).end(),
                         std::back_inserter(val_grad_shape_), [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
    (void)std::transform(input_shapes.at(kSparseAddGradIndex1).begin(), input_shapes.at(kSparseAddGradIndex1).end(),
                         std::back_inserter(x1_indices_shape_), [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
    (void)std::transform(input_shapes.at(kSparseAddGradIndex2).begin(), input_shapes.at(kSparseAddGradIndex2).end(),
                         std::back_inserter(x2_indices_shape_), [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
    (void)std::transform(input_shapes.at(kSparseAddGradIndex3).begin(), input_shapes.at(kSparseAddGradIndex3).end(),
                         std::back_inserter(sum_indices_shape_), [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
    val_grad_size_ = std::accumulate(val_grad_shape_.begin(), val_grad_shape_.end(), 1, std::multiplies{});
    x1_indices_size_ = std::accumulate(x1_indices_shape_.begin(), x1_indices_shape_.end(), 1, std::multiplies{});
    x2_indices_size_ = std::accumulate(x2_indices_shape_.begin(), x2_indices_shape_.end(), 1, std::multiplies{});
    sum_indices_size_ = std::accumulate(sum_indices_shape_.begin(), sum_indices_shape_.end(), 1, std::multiplies{});
    if (val_grad_size_ == 0 || x1_indices_size_ == 0 || x2_indices_size_ == 0 || sum_indices_size_ == 0) {
      MS_LOG(INFO) << "For SparseAddGradSparse, val_grad, x1_indices, x2_indices, sum_indices: " << val_grad_size_
                   << ", " << x1_indices_size_ << ", " << x2_indices_size_ << ", " << sum_indices_size_;
      return -1;
    }

    x1_index_num_ = x1_indices_shape_[0];
    x2_index_num_ = x2_indices_shape_[0];
    sum_index_num_ = sum_indices_shape_[0];
    dim_ = x1_indices_shape_[1];

    input_size_list_.push_back(val_grad_size_ * value_bytes_);
    input_size_list_.push_back(x1_indices_size_ * index_bytes_);
    input_size_list_.push_back(x2_indices_size_ * index_bytes_);
    input_size_list_.push_back(sum_indices_size_ * index_bytes_);
    output_size_list_.push_back(x1_index_num_ * value_bytes_);
    output_size_list_.push_back(x2_index_num_ * value_bytes_);
    work_size_list_.push_back(dim_ * index_bytes_);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    S *dout_ptr = nullptr;
    T *x1_indices_ptr = nullptr;
    T *x2_indices_ptr = nullptr;
    T *out_indices_ptr = nullptr;
    S *dx1_ptr = nullptr;
    S *dx2_ptr = nullptr;
    T *temp_save_ptr = nullptr;
    int flag = GetDeviceAddress<S>(input_ptrs, kSparseAddGradIndex0, kernel_name_, &dout_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kSparseAddGradIndex1, kernel_name_, &x1_indices_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kSparseAddGradIndex2, kernel_name_, &x2_indices_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kSparseAddGradIndex3, kernel_name_, &out_indices_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(output_ptrs, kSparseAddGradIndex0, kernel_name_, &dx1_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(output_ptrs, kSparseAddGradIndex1, kernel_name_, &dx2_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(work_ptrs, kSparseAddGradIndex0, kernel_name_, &temp_save_ptr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream)),
                                       "For SparseAddGrad, cudaStreamSynchronize failed.");
    // call cuda kernel
    MS_LOG(INFO) << "For SparseAddGrad, x1_index_num_, x2_index_num_, sum_index_num_, dim_ " << x1_index_num_ << ", "
                 << x2_index_num_ << ", " << sum_index_num_ << ", " << dim_;
    CalSparseAddGrad(dout_ptr, x1_indices_ptr, x1_index_num_, x2_indices_ptr, x2_index_num_, out_indices_ptr,
                     sum_index_num_, temp_save_ptr, dx1_ptr, dx2_ptr, dim_, device_id_,
                     reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream)),
                                       "For SparseAddGrad, cudaStreamSynchronize failed.");
    return 0;
  }

  void ResetResource() override {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    val_grad_shape_.clear();
    x1_indices_shape_.clear();
    x2_indices_shape_.clear();
    sum_indices_shape_.clear();
  }

 protected:
  int CheckKernelParam() override {
    size_t dim = input_shapes_.at(0).size();
    if (dim != 1) {
      return -1;
    }

    size_t size = input_shapes_.size();
    for (size_t i = 1; i < size; i++) {
      size_t dim = input_shapes_.at(i).size();
      if (dim != kSparseAddGradIndicesDim) {
        return -1;
      }
    }
    return 0;
  }

 private:
  bool is_null_input_ = false;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  size_t val_grad_size_ = 0;
  size_t x1_indices_size_ = 0;
  size_t x2_indices_size_ = 0;
  size_t sum_indices_size_ = 0;
  size_t x1_index_num_ = 0;
  size_t x2_index_num_ = 0;
  size_t sum_index_num_ = 0;
  size_t dim_ = 0;
  size_t index_bytes_ = 0;
  size_t value_bytes_ = 0;
  std::vector<size_t> val_grad_shape_{};
  std::vector<size_t> x1_indices_shape_{};
  std::vector<size_t> x2_indices_shape_{};
  std::vector<size_t> sum_indices_shape_{};
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_ADD_GRAD_HELPER_H_
