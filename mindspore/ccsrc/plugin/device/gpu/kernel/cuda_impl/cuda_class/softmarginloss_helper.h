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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SOFTMARGINLOSS_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SOFTMARGINLOSS_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/softmarginloss_impl.cuh"

namespace mindspore {
namespace cukernel {
class SoftMarginLossAttr : public GpuKernelAttrBase {
 public:
  SoftMarginLossAttr() = default;
  ~SoftMarginLossAttr() override = default;
  string reduction;
};

template <typename T>
class SoftMarginLossHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SoftMarginLossHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    reduction_ = ReductionMode::kMean;
    is_null_softmarginloss_input_ = false;
    input_size_ = 1;
  }

  virtual ~SoftMarginLossHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 2;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    if (input_shapes[0] != input_shapes[1]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input shape should be equal to " << input_shapes[0]
                    << "but got " << input_shapes[1];
      return -1;
    }

    input_shape_ = input_shapes[0];
    input_size_ = 1;
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input_size_ *= input_shape_[i];
    }

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_softmarginloss_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_softmarginloss_input_) {
      return 0;
    }

    T *prediction = nullptr;
    T *target = nullptr;
    T *loss = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &prediction);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &target);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &loss);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    CalSoftMarginLoss(prediction, target, input_size_, reduction_, loss, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<SoftMarginLossAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    reduction_ = kReductionModeMap[attr_ptr_->reduction];
    if (reduction_ != ReductionMode::kMean && reduction_ != ReductionMode::kNone && reduction_ != ReductionMode::kSum) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the 'reduction' must be str and must be in '['none', 'sum', 'mean']',but got "
                    << "'" << reduction_ << "'";
      return -1;
    }
    return 0;
  }

 private:
  ReductionMode reduction_;
  std::shared_ptr<SoftMarginLossAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  bool is_null_softmarginloss_input_;
  size_t input_size_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SOFTMARGINLOSS_HELPER_H_
