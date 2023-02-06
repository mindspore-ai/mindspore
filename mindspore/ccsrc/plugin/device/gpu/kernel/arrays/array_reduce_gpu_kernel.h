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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace kernel {
class ArrayReduceGpuKernelMod : public NativeGpuKernelMod {
 public:
  ArrayReduceGpuKernelMod() { ResetResource(); }
  explicit ArrayReduceGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) { ResetResource(); }
  ~ArrayReduceGpuKernelMod() override { DestroyResource(); }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                                       "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
  }

 protected:
  void ResetResource() {
    cudnn_handle_ = nullptr;
    reduce_tensor_op_ = CUDNN_REDUCE_TENSOR_ADD;
    data_type_ = CUDNN_DATA_FLOAT;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    reduce_indices_ = CUDNN_REDUCE_TENSOR_NO_INDICES;
    reduce_tensor_descriptor_ = nullptr;
    inputA_descriptor_ = nullptr;
    outputC_descriptor_ = nullptr;
    need_skip_execute_ = false;
    keep_dims_ = false;
    all_match_ = false;
    is_null_input_ = false;
    complex_op_type = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    int_op_type = 0;
    kernel_name_ = "ArrayReduce";
    axis_.clear();
  }

  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                        "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
  }

  void InitCudnnResource();
  void InferArrayReduceType();
  void InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape);
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  template <typename T, typename S>
  void LaunchComplexKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using ReduceFunc =
    std::function<bool(ArrayReduceGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;
  ReduceFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> all_any_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> prod_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> sum_list_;
  static std::vector<std::pair<KernelAttr, ReduceFunc>> max_min_mean_list_;
  static std::map<std::string, std::vector<std::pair<KernelAttr, ReduceFunc>>> kernel_attr_list_;

 private:
  cudnnHandle_t cudnn_handle_;
  cudnnReduceTensorOp_t reduce_tensor_op_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t outputC_descriptor_;

  std::vector<int> axis_;
  bool keep_dims_;
  bool need_skip_execute_;
  bool all_match_;
  bool is_null_input_;
  int complex_op_type;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  static constexpr size_t kAxisIndex_{1};
  std::string kernel_type_{"Unknown"};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
