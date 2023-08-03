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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_TENSOR_DENSE_MATMUL_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_TENSOR_DENSE_MATMUL_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_tensor_dense_matmul.cuh"

namespace mindspore {
namespace cukernel {
class SparseTensorDenseMatmulAttr : public GpuKernelAttrBase {
 public:
  SparseTensorDenseMatmulAttr() = default;
  ~SparseTensorDenseMatmulAttr() override = default;
  bool adj_st;
  bool adj_dt;
};

template <typename T, typename S>
class SparseTensorDenseMatmulHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SparseTensorDenseMatmulHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    adj_st_ = false;
    adj_dt_ = false;
    is_null_input_ = false;
  }

  virtual ~SparseTensorDenseMatmulHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 4;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    indices_shape_ = input_shapes[kIndex0];
    values_shape_ = input_shapes[kIndex1];
    dense_shape_ = input_shapes[kIndex3];
    output_shape_ = output_shapes[kIndex0];

    int out_flag =
      CalShapesSizeInBytes<S>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *indices = nullptr;
    S *values = nullptr;
    int64_t *shape = nullptr;
    S *dense = nullptr;
    S *output = nullptr;
    (void)GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &indices);
    (void)GetDeviceAddress<S>(input_ptrs, kIndex1, kernel_name_, &values);
    (void)GetDeviceAddress<int64_t>(input_ptrs, kIndex2, kernel_name_, &shape);
    (void)GetDeviceAddress<S>(input_ptrs, kIndex3, kernel_name_, &dense);
    (void)GetDeviceAddress<S>(output_ptrs, kIndex0, kernel_name_, &output);
    const size_t values_size_ = values_shape_[0];
    const size_t out_dim_1 = output_shape_[1];
    const size_t b_rows = output_shape_[0];
    const size_t b_cols = dense_shape_[1];
    std::vector<T> host_indices(indices_shape_[0] * indices_shape_[1]);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(host_indices.data(), indices, host_indices.size() * sizeof(T), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpy indices from device to host failed.");
    std::vector<int64_t> host_shape(kIndex2);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(host_shape.data(), shape, host_shape.size() * sizeof(int64_t), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpy sparse_shape from device to host failed.");
    for (int32_t pos = 0; pos < indices_shape_[0] * indices_shape_[1]; pos += kIndex2) {
      T row = host_indices[pos];
      T col = host_indices[pos + 1];
      if (row > host_shape[0] || col > host_shape[1]) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                                 << "', the indices including out of bounds index, row range: [0, " << host_shape[0]
                                 << "), col range: [0, " << host_shape << "), but got row: " << row << ", col: " << col;
        return -1;
      }
    }
    std::vector<int64_t> b_shape(dense_shape_);
    if (adj_st_) {
      std::reverse(host_shape.begin(), host_shape.end());
    }
    if (adj_dt_) {
      std::reverse(b_shape.begin(), b_shape.end());
    }
    if (host_shape[1] != b_shape[0]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << ",the second dimension length of 'sparse_shape' "
                    << "must be equal to the first dimension length of 'dense', but got the "
                    << "tensor shape of 'sparse': " << host_shape << ",and the tensor shape of 'dense':" << b_shape;
      return -1;
    }
    auto status = CalSparseTensorDenseMatmul(values_size_, out_dim_1, b_rows, b_cols, indices, values, dense, output,
                                             adj_st_, adj_dt_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }
  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<SparseTensorDenseMatmulAttr>(kernel_attr);
  }

  void ResetResource() noexcept override {
    b_rows = 0;
    b_cols = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 protected:
  int CheckKernelParam() override {
    adj_st_ = attr_ptr_->adj_st;
    adj_dt_ = attr_ptr_->adj_dt;
    if (indices_shape_.size() != kIndex2 || indices_shape_[1] != kIndex2) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' must be a 2-D tensor and"
                    << "the second dimension length must be 2, but got 'indices' shape:" << indices_shape_;
      return -1;
    }
    if (values_shape_.size() != 1 || values_shape_[0] != indices_shape_[0]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'values' must be a 1-D tensor and"
                    << "the first dimension length must be equal to the first dimension length of 'indices',"
                    << "but got 'indices' shape: " << indices_shape_ << ",'values' shape:" << values_shape_;
      return -1;
    }
    return 0;
  }

 private:
  std::shared_ptr<SparseTensorDenseMatmulAttr> attr_ptr_;
  std::vector<int64_t> indices_shape_;
  std::vector<int64_t> values_shape_;
  std::vector<int64_t> dense_shape_;
  std::vector<int64_t> output_shape_;
  size_t b_rows;
  size_t b_cols;
  bool adj_st_{false};
  bool adj_dt_{false};
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SPARSE_TENSOR_DENSE_MATMUL_HELPER_H_
