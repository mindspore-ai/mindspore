/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_TOPK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_TOPK_GPU_KERNEL_H_

#include <limits>
#include <map>
#include <memory>
#include <vector>
#include "infer/topk.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cast_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/topk_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TopKGpuKernelMod : public NativeGpuKernelMod {
 public:
  TopKGpuKernelMod()
      : sorted_(false), is_null_input_(false), outer_size_(1), inner_size_(1), k_(1), input_shape_size_(0) {}
  ~TopKGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspaces,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *indices = GetDeviceAddress<S>(outputs, 1);

    S k_cut = static_cast<S>(k_);

    cudaError_t status = cudaErrorNotReady;
    if (std::is_same<T, half>::value) {
      // remove later! urgent fix for bug: topk has incorrect output for float16
      float init_k = std::numeric_limits<float>::lowest();

      // cast to float32
      float *casted_float32_input = GetDeviceAddress<float>(workspaces, 0);
      float *casted_float32_top_k_output = GetDeviceAddress<float>(workspaces, 1);
      status =
        Cast(outer_size_ * inner_size_, input_addr, casted_float32_input, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);

      // call FastTopK with workspace[n], workspace[n+1] as input, output
      status = FastTopK(outer_size_, inner_size_, casted_float32_input, k_cut, casted_float32_top_k_output, indices,
                        init_k, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);

      // cast workspace[n+1] back to float16
      status =
        Cast(outer_size_ * k_, casted_float32_top_k_output, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      T init_k = std::numeric_limits<T>::lowest();
      status = FastTopK(outer_size_, inner_size_, input_addr, k_cut, output_addr, indices, init_k,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    sorted_ = GetValue<bool>(primitive_->GetAttr("sorted"));
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }

    auto input_shapes = inputs[0]->GetShapeVector();
    auto output_shapes = outputs[0]->GetShapeVector();

    is_null_input_ =
      CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
    if (input_shapes.empty() || is_null_input_) {
      return KRET_OK;
    }

    input_shape_size_ = input_shapes.size();
    outer_size_ = 1;
    for (size_t i = 0; i < input_shapes.size() - 1; i++) {
      outer_size_ *= LongToSize(input_shapes[i]);
    }

    inner_size_ = LongToSizeClipNeg(input_shapes[input_shapes.size() - 1]);
    k_ = LongToSizeClipNeg(output_shapes[output_shapes.size() - 1]);
    InitSizeLists();

    return KRET_OK;
  }

 protected:
  void InitSizeLists() {
    if (std::is_same<T, half>::value) {
      workspace_size_list_.push_back(outer_size_ * inner_size_ * sizeof(float));
      workspace_size_list_.push_back(outer_size_ * k_ * sizeof(float));
    }
  }

 private:
  bool sorted_;
  bool is_null_input_;
  size_t outer_size_;
  size_t inner_size_;
  size_t k_;
  int input_shape_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_TOPK_GPU_KERNEL_H_
