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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ROLL_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ROLL_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roll_impl.cuh"

namespace mindspore {
namespace cukernel {
class RollAttr : public GpuKernelAttrBase {
 public:
  RollAttr() = default;
  ~RollAttr() override = default;
  std::vector<int64_t> axis;
  std::vector<int64_t> shift;
};

template <typename T>
class RollHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit RollHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~RollHelperGpuKernel() = default;
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

    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    dims = static_cast<int64_t>(input_shape_.size());
    size_t outer_size = 1;
    for (int64_t i = input_shape_.size() - 1; i >= 0; i--) {
      outer_size *= input_shape_[i];
    }
    axis = attr_ptr_->axis;
    shift = attr_ptr_->shift;
    T *input = nullptr;
    T *output = nullptr;
    std::vector<int64_t> dim_size;
    std::vector<int64_t> threshold;
    std::vector<int64_t> dim_range;
    std::vector<int64_t> shift_mod_sum;
    std::vector<int64_t> stride;
    std::vector<int64_t> kernel_shift;
    dim_size.resize(dims);
    threshold.resize(dims);
    dim_range.resize(dims);
    shift_mod_sum.resize(dims);
    stride.resize(dims);
    kernel_shift.resize(dims);
    for (unsigned int i = 0; i < axis.size(); i++) {
      int axis_t = axis[i];
      if (axis_t < 0) {
        axis_t += dims;
      }
      const int ds = std::max<int>(input_shape_[axis_t], 1);
      const int sum = shift_mod_sum[axis_t] + shift[i];
      if (ds != 0) {
        shift_mod_sum[axis_t] = (sum % ds + ds) % ds;
      }
    }
    int64_t dim_size_prod = 1;
    for (int i = dims - 1; i >= 0; i--) {
      const int ds = std::max<int>(input_shape_[i], 1);
      dim_size[i] = ds;
      if (ds != 0) {
        threshold[i] = (ds - shift_mod_sum[i]) % ds;
      }
      dim_size_prod *= input_shape_[i];
      dim_range[i] = dim_size_prod;
    }
    for (int i = dims - 1; i >= 0; i--) {
      stride[i] = dim_range[i] / dim_size[i];
      kernel_shift[i] = dim_size[i] - threshold[i];
    }

    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output);
    if (flag != 0) {
      return flag;
    }

    auto status = CalRoll(input, output, &stride[0], &kernel_shift[0], &dim_size[0], outer_size, dims, device_id_,
                          reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }
  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<RollAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    axis = attr_ptr_->axis;
    dims = static_cast<int64_t>(input_shape_.size());
    for (auto s_axis : axis) {
      if (s_axis < -dims || s_axis >= dims) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << ","
                                 << dims << "), but got " << s_axis;
        return -1;
      }
    }
    return 0;
  }

 private:
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> shift;
  std::vector<int64_t> axis;
  std::shared_ptr<RollAttr> attr_ptr_;
  int64_t dims;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ROLL_HELPER_H_
