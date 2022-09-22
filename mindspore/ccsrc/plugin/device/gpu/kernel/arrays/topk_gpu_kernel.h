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
#include <vector>
#include <map>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "ops/topk.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/topk_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TopKGpuKernelMod : public NativeGpuKernelMod {
 public:
  TopKGpuKernelMod()
      : sorted_(false), is_null_input_(false), outer_size_(1), inner_size_(1), k_(1), input_shape_size_(0) {}
  ~TopKGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *k = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *indices = GetDeviceAddress<S>(outputs, 1);

    S k_cut = 0;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&k_cut, k, sizeof(S), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync k_cut failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - TopK");

    if (std::is_same<T, half>::value) {
      // remove later! urgent fix for bug: topk has incorrect output for float16
      float init_k = std::numeric_limits<float>::lowest();

      // cast to float32
      float *casted_float32_input = GetDeviceAddress<float>(workspaces, 0);
      float *casted_float32_top_k_output = GetDeviceAddress<float>(workspaces, 1);
      Cast(outer_size_ * inner_size_, input_addr, casted_float32_input, reinterpret_cast<cudaStream_t>(stream_ptr),
           GET_CTX_DEVICE_ID);

      // call FastTopK with workspace[n], workspace[n+1] as input, output
      FastTopK(outer_size_, inner_size_, casted_float32_input, k_cut, casted_float32_top_k_output, indices, init_k,
               reinterpret_cast<cudaStream_t>(stream_ptr));

      // cast workspace[n+1] back to float16
      Cast(outer_size_ * k_, casted_float32_top_k_output, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr),
           GET_CTX_DEVICE_ID);
    } else {
      T init_k = std::numeric_limits<T>::lowest();
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(&k_cut, k, sizeof(S), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync k_cut failed");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - TopK");
      FastTopK(outer_size_, inner_size_, input_addr, k_cut, output_addr, indices, init_k,
               reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    auto kernel_ptr = std::make_shared<ops::TopK>(base_operator->GetPrim());
    sorted_ = kernel_ptr->get_attr("sorted");
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }

    auto input_shapes = inputs[0]->GetShapeVector();
    auto output_shapes = outputs[0]->GetShapeVector();

    is_null_input_ =
      CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }

    input_shape_size_ = input_shapes.size();
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
