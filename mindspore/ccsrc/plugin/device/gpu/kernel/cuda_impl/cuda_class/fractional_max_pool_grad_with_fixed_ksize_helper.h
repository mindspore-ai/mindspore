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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_max_pool_grad_with_fixed_ksize_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t kDimSize4 = 4;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kFormatNCHWIndexN = 0;
constexpr size_t kFormatNCHWIndexC = 1;
constexpr size_t kFormatNCHWIndexH = 2;
constexpr size_t kFormatNCHWIndexW = 3;
constexpr size_t kInputsNum = 3;
constexpr size_t kOutputsNum = 1;

class FractionalMaxPoolGradWithFixedKsizeAttr : public GpuKernelAttrBase {
 public:
  FractionalMaxPoolGradWithFixedKsizeAttr() = default;
  ~FractionalMaxPoolGradWithFixedKsizeAttr() override = default;
  std::string data_format;
};

template <typename T, typename S>
class FractionalMaxPoolGradWithFixedKsizeHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit FractionalMaxPoolGradWithFixedKsizeHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~FractionalMaxPoolGradWithFixedKsizeHelperGpuKernel() = default;

  int CheckDims() {
    size_t input_dims = origin_input_shape_.size();
    size_t out_backprop_dims = out_backprop_shape_.size();
    size_t argmax_dims = argmax_shape_.size();
    if (!(input_dims == kDimSize4)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input' must be equal to 4, but got "
                    << input_dims << ".";
      return -1;
    }
    if (!(out_backprop_dims == kDimSize4)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'out_backprop' must be equal to 4, but got "
                    << out_backprop_dims << ".";
      return -1;
    }
    if (!(argmax_dims == kDimSize4)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'argmax' must be equal to 4, but got "
                    << argmax_dims << ".";
      return -1;
    }
    return 1;
  }

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    origin_input_shape_ = input_shapes[kInputIndex0];
    out_backprop_shape_ = input_shapes[kInputIndex1];
    argmax_shape_ = input_shapes[kInputIndex2];
    output_shape_ = output_shapes[kOutputIndex0];

    size_t n_dim;
    size_t c_dim;
    size_t h_dim;
    size_t w_dim;
    data_format_ = attr_ptr_->data_format;
    if (data_format_ != "NCHW") {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_format must be NCHW, but got " << data_format_ << ".";
      return -1;
    }
    n_dim = kFormatNCHWIndexN;
    c_dim = kFormatNCHWIndexC;
    h_dim = kFormatNCHWIndexH;
    w_dim = kFormatNCHWIndexW;
    inputN_ = origin_input_shape_[n_dim];
    inputC_ = origin_input_shape_[c_dim];
    inputH_ = origin_input_shape_[h_dim];
    inputW_ = origin_input_shape_[w_dim];
    outputH_ = out_backprop_shape_[h_dim];
    outputW_ = out_backprop_shape_[w_dim];
    int dims_flag = CheckDims();
    if (dims_flag == -1) {
      return dims_flag;
    }

    int inp_flag = 0;
    size_t cur_size_T = sizeof(T);
    for (const auto &val : origin_input_shape_) {
      cur_size_T *= val;
    }
    if (cur_size_T == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size_T);

    cur_size_T = sizeof(T);
    for (const auto &val : out_backprop_shape_) {
      cur_size_T *= val;
    }
    if (cur_size_T == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size_T);

    size_t cur_size_S = sizeof(S);
    for (const auto &val : argmax_shape_) {
      cur_size_S *= val;
    }
    if (cur_size_S == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size_S);

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }

    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *origin_input_ptr = nullptr;
    T *out_backprop_ptr = nullptr;
    S *argmax_ptr = nullptr;
    T *output_ptr = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kInputIndex0, kernel_name_, &origin_input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, kInputIndex1, kernel_name_, &out_backprop_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(input_ptrs, kInputIndex2, kernel_name_, &argmax_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    int64_t dims = static_cast<int64_t>(output_shape_.size());
    int64_t outer_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      outer_size *= output_shape_[i];
    }

    dims = static_cast<int64_t>(out_backprop_shape_.size());
    int64_t out_backprop_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      out_backprop_size *= out_backprop_shape_[i];
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(output_ptr, 0, outer_size * sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream)),
      "In FractionalMaxPoolGradWithFixedKsize kernel, cudaMemsetAsync output variable failed.");
    CalFractionalmaxpoolgradwithfixedksize(origin_input_ptr, out_backprop_ptr, argmax_ptr, output_ptr, outputH_,
                                           outputW_, inputN_, inputC_, inputH_, inputW_, outer_size, out_backprop_size,
                                           device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<FractionalMaxPoolGradWithFixedKsizeAttr>(kernel_attr);
  }

 private:
  std::string data_format_;
  int64_t outputH_{1};
  int64_t outputW_{1};
  int64_t inputN_{1};
  int64_t inputC_{1};
  int64_t inputH_{1};
  int64_t inputW_{1};
  std::shared_ptr<FractionalMaxPoolGradWithFixedKsizeAttr> attr_ptr_;
  std::vector<int64_t> origin_input_shape_;
  std::vector<int64_t> out_backprop_shape_;
  std::vector<int64_t> argmax_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
