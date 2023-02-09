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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_MEDIAN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_MEDIAN_GPU_KERNEL_H_

#include <vector>
#include <map>
#include "mindspore/core/ops/median.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/median_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMedianInputsNum = 1;
constexpr size_t kMedianOutputsNum = 2;
template <typename T, typename S>
class MedianGpuKernelMod : public NativeGpuKernelMod {
 public:
  MedianGpuKernelMod() : global_median_(false), keep_dims_(false), axis_(0) {}
  ~MedianGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output0_addr = GetDeviceAddress<T>(outputs, 0);
    S *output1_addr = nullptr;
    if (!global_median_) {
      output1_addr = GetDeviceAddress<S>(outputs, 1);
    }
    Median(input_addr, output0_addr, output1_addr, input_shape_, axis_, global_median_,
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    auto kernel_ptr = std::dynamic_pointer_cast<ops::Median>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' cast Median ops failed!";
      return false;
    }
    if (inputs.size() != kMedianInputsNum || outputs.size() > kMedianOutputsNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size should be " << kMedianInputsNum << " and "
                    << kMedianOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
      return false;
    }
    global_median_ = kernel_ptr->get_global_median();
    keep_dims_ = kernel_ptr->get_keep_dims();
    attr_axis_ = kernel_ptr->get_axis();
    return true;
  }

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
    input_shapes.emplace_back(inp_shape);
    std::vector<size_t> input_size_list;
    int inp_flag =
      cukernel::CalShapesSizeInBytes<T>(input_shapes, kMedianInputsNum, kernel_name_, "input_shapes", &input_size_list);
    if (inp_flag == -1) {
      return KRET_RESIZE_FAILED;
    }
    is_null_input_ = inp_flag == 1;
    axis_ = attr_axis_;
    input_shape_ = inputs[0]->GetShapeVector();
    if (global_median_) {
      int input_size = 1;
      for (size_t i = 0; i < input_shape_.size(); i++) {
        input_size *= input_shape_[i];
      }
      input_shape_.clear();
      input_shape_.push_back(input_size);
      if (axis_ != 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', when 'global_median' is True, the 'axis' must be 0, but got "
                      << axis_;
        return KRET_RESIZE_FAILED;
      }
      if (keep_dims_) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', when 'global_median' is True, the 'keep_dims' must be False, but got " << keep_dims_;
        return KRET_RESIZE_FAILED;
      }
    }
    int64_t dims = static_cast<int64_t>(input_shape_.size());
    if (dims == 0) {
      if (axis_ < -1 || axis_ > 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-1,1), but got " << axis_;
        return KRET_RESIZE_FAILED;
      }
    } else if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                    << "), but got " << axis_;
      return KRET_RESIZE_FAILED;
    }
    if (axis_ < 0) {
      if (dims == 0) {
        axis_ = 0;
      } else {
        axis_ += dims;
      }
    }
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  bool global_median_;
  bool keep_dims_;
  int64_t attr_axis_;
  int64_t axis_;
  std::vector<int64_t> input_shape_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_MEDIAN_GPU_KERNEL_H_
