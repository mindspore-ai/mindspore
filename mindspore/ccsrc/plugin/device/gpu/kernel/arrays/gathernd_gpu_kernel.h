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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
#include <map>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/gather_nd.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherNdFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherNdFwdGpuKernelMod() = default;
  ~GatherNdFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *dev_batch_strides = GetDeviceAddress<S>(workspace, 0);
    S *dev_batch_indices = GetDeviceAddress<S>(workspace, 1);
    int *device_flag = GetDeviceAddress<int>(workspace, 2);
    const size_t strides_len = sizeof(S) * batch_strides_.size();
    const size_t indices_len = sizeof(S) * batch_indices_.size();
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dev_batch_strides, &batch_strides_[0], strides_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in GatherNdFwdGpuKernelMod::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dev_batch_indices, &batch_indices_[0], indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in GatherNdFwdGpuKernelMod::Launch.");

    auto ret = GatherNd(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], dev_batch_strides,
                        dev_batch_indices, device_flag, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (ret.first >= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the indices[" << ret.first << "]: " << ret.second
                        << ", does not index into input_shape: " << input_shapes_ << ".";
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(base_operator);
    const size_t input_num = 2;
    const size_t output_num = 1;
    kernel_name_ = base_operator->GetPrim()->name();
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    input_shapes_ = inputs[0]->GetShapeVector();
    ShapeVector indices_shapes = inputs[1]->GetShapeVector();
    // make a scalar to tensor whose shape is (1,)
    if (indices_shapes.size() == 0) {
      indices_shapes.emplace_back(1);
    }
    int64_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes.size() - IntToSize(1); i++) {
      dim_of_indices *= indices_shapes[i];
    }

    int64_t dim_after_indices = 1;
    size_t dim_indices_last = indices_shapes[indices_shapes.size() - IntToSize(1)];
    for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
      dim_after_indices *= input_shapes_[i];
    }
    dims_ = {LongToSize(dim_of_indices), LongToSize(dim_after_indices), dim_indices_last};

    batch_strides_.resize(dim_indices_last, 0);
    batch_indices_.resize(dim_indices_last, 0);

    if (dim_indices_last > 0) {
      batch_strides_[dim_indices_last - 1] = input_shapes_[dim_indices_last - 1];
      batch_indices_[dim_indices_last - 1] = dims_[1];
    }
    for (size_t i = dim_indices_last - 1; i > 0; --i) {
      batch_strides_[i - 1] = input_shapes_[i - 1];
      batch_indices_[i - 1] = batch_indices_[i] * input_shapes_[i];
    }
    const size_t strides_size = sizeof(S) * batch_strides_.size();
    const size_t indices_size = sizeof(S) * batch_indices_.size();
    workspace_size_list_ = {strides_size, indices_size, sizeof(int)};
    return ret;
  }

 private:
  std::vector<size_t> dims_;
  std::vector<int64_t> input_shapes_;
  std::vector<S> batch_strides_;
  std::vector<S> batch_indices_;
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_ARRAYS_GATHERND_GPU_KERNEL_H_
