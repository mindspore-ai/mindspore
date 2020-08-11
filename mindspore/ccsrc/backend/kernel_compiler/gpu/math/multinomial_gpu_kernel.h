/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/multinomial_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/float_status_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/cumsum_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class MultinomialGpuKernel : public GpuKernel {
 public:
  MultinomialGpuKernel() : input_size_0_(0), output_size_(0), distributions_(0), workspace_size_(sizeof(curandState)) {}
  ~MultinomialGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
    curandState *devStates = reinterpret_cast<curandState *>(workspace_addr);
    int *output_addr = GetDeviceAddress<int>(outputs, 0);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    int categories = SizeToInt(inputs[0]->size / sizeof(T)) / distributions_;
    int num_sample = SizeToInt(outputs[0]->size / sizeof(T)) / distributions_;
    // check input
    T *flag = nullptr;
    T *cflag = nullptr;
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMalloc(reinterpret_cast<void **>(&cflag), sizeof(T)), "cudaMalloc failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMallocHost(&flag, sizeof(T)), "cudaMallocHost failed.");
    CalFloatStatus(input_size_0_ / sizeof(T), input_addr, cflag, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpy(flag, cflag, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpyAsync failed.");
    if (*flag > 0) {
      MS_LOG(EXCEPTION) << "Input is invalid (containing NaN, -inf or inf)";
    }
    CheckNonNeg(input_size_0_ / sizeof(T), input_addr, cflag, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpy(flag, cflag, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpyAsync failed.");
    if (*flag > 0) {
      MS_LOG(EXCEPTION) << "Input is invalid (input element < 0)";
    }
    T *cum_sum_input = nullptr;
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMalloc(reinterpret_cast<void **>(&cum_sum_input), input_size_0_),
                               "cudaMalloc failed.");
    CumSum(input_addr, cum_sum_input, cum_sum_input, IntToSize(distributions_), IntToSize(categories), 1,
           IntToSize(categories), 1, false, false, reinterpret_cast<cudaStream_t>(stream_ptr));
    CheckZero(IntToSize(distributions_), IntToSize(categories), cum_sum_input, cflag,
              reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpy(flag, cflag, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpyAsync failed.");
    if (*flag > 0) {
      MS_LOG(EXCEPTION) << "Input is invalid (sum <= 0)";
    }
    CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaStreamSynchronize failed.");
    Multinomial(seed_, cum_sum_input, num_sample, devStates, output_addr, IntToSize(distributions_),
                IntToSize(categories), reinterpret_cast<cudaStream_t>(stream_ptr));

    CHECK_CUDA_RET_WITH_EXCEPT(cudaFree(cum_sum_input), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFree(cflag), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFreeHost(flag), "cudaFreeHost failed.");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but multinomial needs 2 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but multinomial needs 1 output.";
      return false;
    }
    auto input_shape_0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape_0.size() == 1) {
      distributions_ = 1;
    } else {
      distributions_ = input_shape_0[0];
    }
    input_size_0_ = sizeof(T);
    for (size_t i = 0; i < input_shape_0.size(); i++) {
      input_size_0_ *= input_shape_0[i];
    }
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    output_size_ = sizeof(int);
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
      workspace_size_ *= output_shape[i];
    }
    seed_ = GetValue<int>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("seed"));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_0_);
    input_size_list_.push_back(sizeof(int));
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  size_t input_size_0_;
  size_t output_size_;
  size_t distributions_;
  size_t workspace_size_;
  int seed_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MULTINOMIAL_GPU_KERNEL_H_
