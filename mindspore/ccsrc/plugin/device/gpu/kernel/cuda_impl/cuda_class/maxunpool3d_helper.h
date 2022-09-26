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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MAXUNPOOL3D_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MAXUNPOOL3D_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxunpool3d_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxunpool3d_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
class MaxUnpool3DAttr : public GpuKernelAttrBase {
 public:
  MaxUnpool3DAttr() = default;
  ~MaxUnpool3DAttr() override = default;
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_shape;
  std::string data_format;
};

template <typename T, typename S>
class MaxUnpool3DHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit MaxUnpool3DHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~MaxUnpool3DHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    input_shape_ = input_shapes[kIndex0];
    indices_shape_ = input_shapes[kIndex1];

    size_t cur_size_T = sizeof(T);
    for (const auto &val : input_shape_) {
      cur_size_T *= val;
    }
    input_size_list_.emplace_back(cur_size_T);

    size_t cur_size_S = sizeof(S);
    for (const auto &val : indices_shape_) {
      cur_size_S *= val;
    }
    input_size_list_.emplace_back(cur_size_S);
    work_size_list_.emplace_back(sizeof(int64_t));
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    output_shape_ = output_shapes[kIndex0];
    is_null_input_ = (out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    S *indices = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, kIndex1, kernel_name_, &indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    int64_t dims = static_cast<int64_t>(input_shape_.size());
    int64_t outer_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      outer_size *= output_shape_[i];
    }

    int64_t thread_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      thread_size *= input_shape_[i];
    }
    CalMaxUnpool3D(input_ptr, indices, input_shape_, output_shape_, output_ptr, outer_size, thread_size, data_format_,
                   device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<MaxUnpool3DAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    data_format_ = attr_ptr_->data_format;
    if (data_format_ != "NCDHW" && data_format_ != "NDHWC") {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'data_format' must be 'NCDHW' or 'NDHWC' ,"
                    << " but got " << data_format_;
      return -1;
    }
    data_format_ = attr_ptr_->data_format;
    return 0;
  }

 private:
  std::shared_ptr<MaxUnpool3DAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> indices_shape_;
  std::vector<int64_t> output_shape_;
  std::string data_format_;
  bool is_null_input_;
};

class MaxUnpool3DGradAttr : public GpuKernelAttrBase {
 public:
  MaxUnpool3DGradAttr() = default;
  ~MaxUnpool3DGradAttr() override = default;
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_shape;
  std::string data_format;
};

template <typename T, typename S>
class MaxUnpool3DGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit MaxUnpool3DGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~MaxUnpool3DGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    backprop_input_shape_ = input_shapes[kIndex0];
    grad_shape_ = input_shapes[kIndex1];
    indices_shape_ = input_shapes[kIndex2];

    size_t cur_size_T = sizeof(T);
    for (const auto &val : backprop_input_shape_) {
      cur_size_T *= val;
    }
    input_size_list_.emplace_back(cur_size_T);

    cur_size_T = sizeof(T);
    for (const auto &val : grad_shape_) {
      cur_size_T *= val;
    }
    input_size_list_.emplace_back(cur_size_T);

    size_t cur_size_S = sizeof(S);
    for (const auto &val : indices_shape_) {
      cur_size_S *= val;
    }
    input_size_list_.emplace_back(cur_size_S);
    work_size_list_.emplace_back(sizeof(int64_t));
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    backprop_output_shape_ = output_shapes[kIndex0];
    is_null_input_ = (out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    T *grad = nullptr;
    S *indices = nullptr;
    T *output_ptr = nullptr;
    // int64_t *gpuflag = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, kIndex0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kIndex1, kernel_name_, &grad);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, kIndex2, kernel_name_, &indices);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kIndex0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    int64_t dims = static_cast<int64_t>(backprop_input_shape_.size());
    int64_t outer_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      outer_size *= backprop_output_shape_[i];
    }
    CalMaxUnpool3DGrad(grad, indices, backprop_input_shape_, grad_shape_, output_ptr, outer_size, grad_data_format_,
                       device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<MaxUnpool3DGradAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    grad_data_format_ = attr_ptr_->data_format;
    if (grad_data_format_ != "NCDHW" && grad_data_format_ != "NDHWC") {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'data_format' must be 'NCDHW' or 'NDHWC' ,"
                    << " but got " << grad_data_format_;
      return -1;
    }
    grad_data_format_ = attr_ptr_->data_format;
    return 0;
  }

 private:
  std::shared_ptr<MaxUnpool3DGradAttr> attr_ptr_;
  std::vector<int64_t> backprop_input_shape_;
  std::vector<int64_t> grad_shape_;
  std::vector<int64_t> indices_shape_;
  std::vector<int64_t> backprop_output_shape_;
  std::string grad_data_format_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ARGMAX_HELPER_H_
