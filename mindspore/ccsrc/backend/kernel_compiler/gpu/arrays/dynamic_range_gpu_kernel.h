/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/dynamic_range_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DynamicRangeGpuKernel : public GpuKernel {
 public:
  DynamicRangeGpuKernel() { ResetResource(); }
  ~DynamicRangeGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

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

    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(&error_code, error_code_device_address, sizeof(DynamicRangeErrorCode),
                                              cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Failed to copy error code to host.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed");

    // use workspace[0] for actual output shape, we know it must be 1d
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(&output_shape_, output_shape_device_address, sizeof(int64_t),
                                              cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "Failed to copy output_shape to host.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed");

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

  void PostExecute() override {
    // required synchronize for PostExecute
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                               "cudaStreamSynchronize failed");

    std::vector<TypeId> output_type = {AnfAlgo::GetOutputInferDataType(kernel_node_.lock(), 0)};
    std::vector<std::vector<size_t>> output_shape = {{(size_t)output_shape_}};
    AnfAlgo::SetOutputInferTypeAndShape(output_type, output_shape, kernel_node_.lock().get());
  }

  void ResetResource() noexcept override {
    stream_ptr_ = nullptr;
    output_shape_ = 0;
    max_output_length_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 3) {
      MS_LOG(ERROR) << input_count << " inputs were provided, but DynamicRangeGpuKernel expects 3.";
      return false;
    }

    max_output_length_ = GetAttr<int64_t>(kernel_node, "maxlen");
    kernel_node_ = kernel_node;
    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(max_output_length_ * sizeof(T));

    // this op outputs a 1d tensor, size of one int64_t is enough space to hold the shape.
    workspace_size_list_.push_back(sizeof(int64_t));
    workspace_size_list_.push_back(sizeof(DynamicRangeErrorCode));
    return;
  }

 private:
  void *stream_ptr_;
  int64_t output_shape_;
  int64_t max_output_length_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_RANGE_GPU_KERNEL_H_
