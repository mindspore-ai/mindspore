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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PAD_V3_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PAD_V3_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "ops/op_name.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_v3_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr int64_t kPaddingSize = 2;
constexpr int64_t kPad1D = 2;
constexpr int64_t kPad2D = 4;
constexpr int64_t kPad3D = 6;
constexpr size_t kMaxDim = 5;
constexpr size_t kMaxPadDim = 3;

class PadV3Attr : public GpuKernelAttrBase {
 public:
  PadV3Attr() = default;
  ~PadV3Attr() override = default;
  bool paddings_contiguous;
  std::string mode;
  std::vector<int64_t> paddings;
};

template <typename T, typename S>
class PadV3HelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit PadV3HelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    paddings_contiguous_ = 0;
    is_null_input_ = false;
  }

  virtual ~PadV3HelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    size_t x_size = sizeof(T);
    for (auto val : input_shapes[0]) {
      x_size *= val;
    }
    size_t pad_size = sizeof(S);
    for (auto val : input_shapes[1]) {
      pad_size *= val;
    }
    input_size_list_.emplace_back(x_size);
    input_size_list_.emplace_back(pad_size);
    if (attr_ptr_->mode == ops::kConstant) {
      input_size_list_.emplace_back(sizeof(T));
    }
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    input_size_ = input_size_list_[0] / sizeof(T);
    output_size_ = output_size_list_[0] / sizeof(T);
    is_null_input_ = (out_flag == 1);
    if (input_shapes[0].size() > kMaxDim) {
      MS_EXCEPTION(ValueError) << "For PadV3 GPU, the max dim is " << kMaxDim << ", but got " << input_shapes[0].size();
    }
    size_t expand_dim = kMaxDim - input_shapes[0].size();
    for (size_t i = 0; i < expand_dim; ++i) {
      input_shape_5d_.emplace_back(1);
      output_shape_5d_.emplace_back(1);
    }
    for (size_t i = 0; i < input_shapes[0].size(); ++i) {
      input_shape_5d_.emplace_back(input_shapes[0][i]);
      output_shape_5d_.emplace_back(output_shapes[0][i]);
    }

    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    if (mode_ == ops::kConstant) {
      T *constant_value = nullptr;
      flag = GetDeviceAddress<T>(input_ptrs, 2, kernel_name_, &constant_value);
      if (flag != 0) {
        return flag;
      }
      CalConstantPad3d(output_size_, input_ptr, input_shape_5d_[0], input_shape_5d_[1], input_shape_5d_[2],
                       input_shape_5d_[3], input_shape_5d_[4], output_shape_5d_[2], output_shape_5d_[3],
                       output_shape_5d_[4], paddings_3d_[0].first, paddings_3d_[1].first, paddings_3d_[2].first,
                       constant_value, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    } else if (mode_ == ops::kReflect) {
      CalReflectPad3d(output_size_, input_ptr, input_shape_5d_[0], input_shape_5d_[1], input_shape_5d_[2],
                      input_shape_5d_[3], input_shape_5d_[4], output_shape_5d_[2], output_shape_5d_[3],
                      output_shape_5d_[4], paddings_3d_[0].first, paddings_3d_[1].first, paddings_3d_[2].first,
                      output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    } else if (mode_ == ops::kEdge) {
      CalEdgePad3d(output_size_, input_ptr, input_shape_5d_[0], input_shape_5d_[1], input_shape_5d_[2],
                   input_shape_5d_[3], input_shape_5d_[4], output_shape_5d_[2], output_shape_5d_[3],
                   output_shape_5d_[4], paddings_3d_[0].first, paddings_3d_[1].first, paddings_3d_[2].first, output_ptr,
                   device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    } else if (mode_ == ops::kCircular) {
      CalCircularPad3d(output_size_, input_ptr, input_shape_5d_[kIndex2], input_shape_5d_[kIndex3],
                       input_shape_5d_[kIndex4], output_shape_5d_[kIndex2], output_shape_5d_[kIndex3],
                       output_shape_5d_[kIndex4], paddings_3d_[kIndex0].first, paddings_3d_[kIndex1].first,
                       paddings_3d_[kIndex2].first, paddings_3d_[kIndex0].second, paddings_3d_[kIndex1].second,
                       paddings_3d_[kIndex2].second, output_ptr, device_id_,
                       reinterpret_cast<cudaStream_t>(cuda_stream));
    }
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<PadV3Attr>(kernel_attr);
  }
  void ResetResource() override {
    input_shape_5d_.clear();
    output_shape_5d_.clear();
    paddings_3d_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 protected:
  int CheckKernelParam() override {
    paddings_switched_ = attr_ptr_->paddings;
    mode_ = attr_ptr_->mode;

    for (int64_t i = kMaxPadDim; i > 0; --i) {
      int64_t before, after;
      if (i > SizeToLong(paddings_switched_.size()) / kPaddingSize) {
        before = 0;
        after = 0;
      } else {
        before = paddings_switched_[(i - 1) * kPaddingSize];
        after = paddings_switched_[i * kPaddingSize - 1];
      }
      if (mode_ == ops::kReflect) {
        int64_t output_size = input_shape_5d_[kMaxDim - i];
        if (before >= output_size || after >= output_size) {
          MS_EXCEPTION(ValueError)
            << "For 'PadV3' of 'reflect' mode, paddings must be less than the input dimension size: " << before
            << ", but " << after << " not less than " << output_size;
        }
      }
      (void)paddings_3d_.emplace_back(std::make_pair(before, after));
    }

    return 0;
  }

 private:
  std::string mode_;
  bool paddings_contiguous_;
  std::shared_ptr<PadV3Attr> attr_ptr_;
  std::vector<int64_t> paddings_switched_;
  size_t input_size_;
  size_t output_size_;
  std::vector<std::pair<int64_t, int64_t>> paddings_3d_;
  std::vector<int64_t> input_shape_5d_;
  std::vector<int64_t> output_shape_5d_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_PAD_V3_HELPER_H_
