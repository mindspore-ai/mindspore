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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_RANGE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_RANGE_GPU_KERNEL_H_

#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dynamic_range_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "mindspore/core/ops/range.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputSize = 3;

template <typename T>
class RangeGpuKernelMod : public NativeGpuKernelMod {
 public:
  RangeGpuKernelMod() { ResetResource(); }
  ~RangeGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *range_start = GetDeviceAddress<T>(inputs, 0);
    T *range_end = GetDeviceAddress<T>(inputs, 1);
    T *range_delta = GetDeviceAddress<T>(inputs, 2);
    T *output_device_address = GetDeviceAddress<T>(outputs, 0);
    int64_t *output_shape_device_address = GetDeviceAddress<int64_t>(workspace, 0);
    DynamicRangeErrorCode *error_code_device_address = GetDeviceAddress<DynamicRangeErrorCode>(workspace, 1);

    stream_ptr_ = stream_ptr;

    CudaValidateInputAndInferShape(range_start, range_end, range_delta, output_shape_device_address,
                                   error_code_device_address, max_output_length_,
                                   reinterpret_cast<cudaStream_t>(stream_ptr));

    DynamicRangeErrorCode error_code = DynamicRangeErrorCode::kOk;

    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(&error_code, error_code_device_address, sizeof(DynamicRangeErrorCode), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Failed to copy error code to host.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed");

    // use workspace[0] for actual output shape, we know it must be 1d
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(&output_shape_, output_shape_device_address, sizeof(int64_t), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Failed to copy output_shape to host.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed");

    LogExceptionIfNotOk(error_code);

    CalRange(range_start, range_end, range_delta, output_device_address, output_shape_device_address,
             error_code_device_address, max_output_length_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  void LogExceptionIfNotOk(DynamicRangeErrorCode error_code) {
    switch (error_code) {
      case DynamicRangeErrorCode::kOk:
        return;
      case DynamicRangeErrorCode::kDeltaIsZero:
        MS_LOG(EXCEPTION) << "gpu RangeOp input error: delta cannot be equal to zero";
        break;
      case DynamicRangeErrorCode::kInvalidPositiveDelta:
        MS_LOG(EXCEPTION) << "gpu RangeOp input error: delta cannot be positive when limit < start";
        break;
      case DynamicRangeErrorCode::kInvalidNegativeDelta:
        MS_LOG(EXCEPTION) << "gpu RangeOp input error: delta cannot be negative when limit > start";
        break;
      case DynamicRangeErrorCode::kMaxSizeExceeded:
        MS_LOG(EXCEPTION) << "gpu RangeOp memory error: the number of elements in the output exceeds maxlen";
        break;
      default:
        MS_LOG(EXCEPTION) << "gpu RangeOp unknown error";
    }
  }

  void ResetResource() noexcept {
    stream_ptr_ = nullptr;
    output_shape_ = 0;
    max_output_length_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_name = base_operator->GetPrim()->name();
    size_t input_count = inputs.size();
    if (input_count != kInputSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs must be 3, but got " << input_count;
    }

    auto kernel_ptr = std::make_shared<ops::Range>(base_operator->GetPrim());

    max_output_length_ = kernel_ptr->get_maxlen();
    outputs_ = outputs;
    is_need_retrieve_output_shape_ = true;
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    auto ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    workspace_size_list_.push_back(sizeof(int64_t));
    workspace_size_list_.push_back(sizeof(DynamicRangeErrorCode));
    return KRET_OK;
  }

 protected:
  void SyncData() override {
    // required synchronize for UpdateOp
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                       "cudaStreamSynchronize failed");
    std::vector<int64_t> output_shape = {output_shape_};
    outputs_[0]->SetShapeVector(output_shape);
  }

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  void *stream_ptr_;
  int64_t output_shape_;
  int64_t max_output_length_;
  std::vector<KernelTensorPtr> outputs_{};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_RANGE_GPU_KERNEL_H_
