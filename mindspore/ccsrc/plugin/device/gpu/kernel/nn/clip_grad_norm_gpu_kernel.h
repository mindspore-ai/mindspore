/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CLIP_GRAD_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CLIP_GRAD_NORM_GPU_KERNEL_H_

#include <string>
#include <vector>
#include <algorithm>
#include "utils/log_adapter.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/clip_grad_norm_impl.cuh"

namespace mindspore::kernel {
constexpr size_t kArgMaxDim = 7;

template <typename T>
class ClipGradNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  ClipGradNormGpuKernelMod()
      : cudnn_handle_(nullptr),
        data_type_(CUDNN_DATA_FLOAT),
        nan_prop_(CUDNN_NOT_PROPAGATE_NAN),
        reduce_indices_(CUDNN_REDUCE_TENSOR_NO_INDICES),
        reduce_tensor_descriptor_(nullptr),
        input_descriptor_(nullptr),
        output_descriptor_(nullptr),
        all_match_(false),
        is_null_input_(false),
        x_size_(0),
        clip_norm_size_(0),
        scaling_factor_size_(0),
        output_size_(0),
        workspace_size_(0) {}

  ~ClipGradNormGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    // Get address
    constexpr size_t input_num_expected = 3;
    constexpr size_t workspace_num_expected = 3;
    MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == input_num_expected, "Size not equal");
    MS_EXCEPTION_IF_CHECK_FAIL(workspace.size() == workspace_num_expected, "Size not equal");
    MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == 1, "Size not equal");
    constexpr size_t scaling_factor_index = 2;
    constexpr size_t reduce_out_index = 2;
    T *x_addr = GetDeviceAddress<T>(inputs, 0);
    T *clip_norm_addr = GetDeviceAddress<T>(inputs, 1);
    float *scaling_factor_addr = GetDeviceAddress<float>(inputs, scaling_factor_index);
    float *scaling_out_addr = GetDeviceAddress<float>(workspace, 0);
    float *reduce_workspace_addr = GetPossiblyNullDeviceAddress<float>(workspace, 1);
    float *reduce_out_addr = GetDeviceAddress<float>(workspace, reduce_out_index);
    float *output_addr = GetDeviceAddress<float>(outputs, 0);

    // Run gradient tensor scaling.
    ScalingGradOp(x_size_ / sizeof(T), x_addr, scaling_factor_addr, scaling_out_addr,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    // Run reduce sum operation(keep_dims=True) for gradient tensor.
    constexpr size_t alpha = 1;
    constexpr size_t beta = 0;
    if (all_match_) {
      CHECK_CUDA_RET_WITH_EXCEPT(
        kernel_node_,
        cudaMemcpyAsync(reduce_out_addr, scaling_out_addr, workspace_size_list_[reduce_out_index],
                        cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync for 'ClipGradNormGpuKernelMod' failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          workspace_size_list_[1], &alpha, input_descriptor_, scaling_out_addr, &beta,
                          output_descriptor_, reduce_out_addr),
        "cudnnReduceTensor for 'ClipGradNormGpuKernelMod' failed");
    }
    // Update gradient tensor by argument 'clip_norm'
    ClipGradNormOp(output_size_ / sizeof(float), scaling_out_addr, clip_norm_addr, reduce_out_addr, output_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_CHECK_FAIL(kernel_name == "ClipGradNorm", "Kernel name is not ClipGradNorm");
    kernel_node_ = kernel_node;
    // Init resource for cudnnreducetensor operation.
    InitResource();
    if (!CheckIONumber(kernel_node)) {
      return false;
    }
    // Check input and output shape
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    size_t input_dim = input_shape.size();
    if (!CheckValidShape(input_shape, output_shape, input_dim)) {
      return true;
    }
    // Init member variables.
    InitAxis(kernel_node, output_shape, SizeToInt(input_dim));
    clip_norm_size_ = sizeof(T);
    scaling_factor_size_ = sizeof(float);
    x_size_ = sizeof(T);
    output_size_ = sizeof(float);
    std::for_each(output_shape.begin(), output_shape.end(), [this](const size_t &v) {
      x_size_ *= v;
      output_size_ *= v;
    });
    InitShapeInfo(input_shape, output_shape);
    // Determine the reduce operation.
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }

  void InitSizeLists() override {
    input_size_list_.emplace_back(x_size_);
    input_size_list_.emplace_back(clip_norm_size_);
    input_size_list_.emplace_back(scaling_factor_size_);
    output_size_list_.emplace_back(output_size_);
    // Init workspace size for gradient tensor scaling calculate.
    workspace_size_list_.emplace_back(output_size_);
    // Init workspace size for gradient tensor reduce sum calculate.
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_,
                                                               input_descriptor_, output_descriptor_, &workspace_size_),
                                "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.emplace_back(workspace_size_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(output_descriptor_, &workspace_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.emplace_back(workspace_size_);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }

  bool CheckIONumber(const CNodePtr &kernel_node) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    constexpr size_t input_num_expected = 3;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != input_num_expected) {
      MS_LOG(ERROR) << "The input number of kernel node [" << kernel_node->DebugString() << "] should be "
                    << input_num_expected << ", but got " << input_num;
      return false;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "The output number of kernel node [" << kernel_node->DebugString() << "] should be " << 1
                    << ", but got " << output_num;
      return false;
    }
    return true;
  }

  bool CheckValidShape(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape,
                       size_t input_dim) {
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ClipGradNormGpuKernelMod', input or output is null.";
      InitSizeLists();
      return false;
    }
    if (input_shape.size() != output_shape.size()) {
      MS_LOG(EXCEPTION) << "The size of input shape: " << input_shape.size()
                        << " and the size of output shape: " << output_shape.size() << " are different.";
    }
    if (input_dim > kArgMaxDim) {
      MS_LOG(EXCEPTION) << "Broadcast operation is not supported when dim exceeds than " << kArgMaxDim;
    }
    CheckTensorSize({input_shape, output_shape});
    return true;
  }

  void InitAxis(const CNodePtr &kernel_node, const std::vector<size_t> &output_shape, int input_dim) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->GetAttr("axis")->isa<ValueTuple>() || prim->GetAttr("axis")->isa<ValueList>()) {
      std::vector<int64_t> attr_axis = GetAttr<std::vector<int64_t>>(kernel_node, "axis");
      if (!attr_axis.empty()) {
        std::vector<int> attr_axis_int;
        (void)std::transform(attr_axis.begin(), attr_axis.end(), std::back_inserter(attr_axis_int),
                             [](const int64_t &v) { return LongToInt(v); });
        for (const auto &v : attr_axis_int) {
          v < 0 ? axis_.emplace_back(v + input_dim) : axis_.emplace_back(v);
        }
        std::sort(axis_.begin(), axis_.end());
        auto multiple_ops = std::unique(axis_.begin(), axis_.end());
        (void)axis_.erase(multiple_ops, axis_.end());
      }
    } else if (prim->GetAttr("axis")->isa<Int64Imm>()) {
      int axis = LongToInt(GetAttr<int64_t>(kernel_node, "axis"));
      axis < 0 ? axis_.emplace_back(axis + input_dim) : axis_.emplace_back(axis);
    } else {
      MS_LOG(EXCEPTION) << "The attribute axis type is invalid.";
    }

    bool exceed_bound =
      std::any_of(axis_.begin(), axis_.end(), [&input_dim](const int &v) { return v < 0 || v >= input_dim; });
    if (exceed_bound) {
      MS_LOG(EXCEPTION) << "For 'ClipGradNormGpuKernelMod', the value of axis should be in range of [-" << input_dim
                        << ", " << (input_dim - 1) << "].";
    }
  }

  void InitShapeInfo(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
    // Determine which dimension will be reduced.
    std::vector<size_t> reduce_output_shape = output_shape;
    if (axis_.empty()) {
      std::fill(reduce_output_shape.begin(), reduce_output_shape.end(), 1);
    } else {
      std::for_each(axis_.begin(), axis_.end(), [&reduce_output_shape](const int &v) { reduce_output_shape[v] = 1; });
    }
    // Whether is all matched.
    all_match_ = true;
    input_shape_.resize(kArgMaxDim, 1);
    output_shape_.resize(kArgMaxDim, 1);
    reduce_output_shape_.resize(kArgMaxDim, 1);
    for (size_t i = 0; i < output_shape.size(); ++i) {
      input_shape_[i] = input_shape[i];
      output_shape_[i] = output_shape[i];
      reduce_output_shape_[i] = reduce_output_shape[i];
      if (input_shape_[i] != reduce_output_shape_[i]) {
        all_match_ = false;
      }
    }
    // Infer input and output descriptor.
    InferInAndOutDesc(input_shape, reduce_output_shape);
  }

  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &reduce_output_shape) {
    constexpr size_t split_dim = 4;
    constexpr size_t dim_idx_two = 2;
    constexpr size_t dim_idx_three = 3;
    if (input_shape.size() <= split_dim) {
      std::vector<size_t> new_input_shape;
      ShapeNdTo4d(input_shape, &new_input_shape);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, data_type_, new_input_shape[0],
                                   new_input_shape[1], new_input_shape[dim_idx_two], new_input_shape[dim_idx_three]),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, input_descriptor_, data_type_, kernel_node_);
    }
    if (reduce_output_shape.size() <= split_dim) {
      std::vector<size_t> new_reduce_output_shape;
      ShapeNdTo4d(reduce_output_shape, &new_reduce_output_shape);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, data_type_, new_reduce_output_shape[0],
                                   new_reduce_output_shape[1], new_reduce_output_shape[dim_idx_two],
                                   new_reduce_output_shape[dim_idx_three]),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(reduce_output_shape, output_descriptor_, data_type_, kernel_node_);
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;

  bool all_match_{false};
  bool is_null_input_{false};
  size_t x_size_;
  size_t clip_norm_size_;
  size_t scaling_factor_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<int> axis_;

  // Used for broadcast operation.
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> reduce_output_shape_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CLIP_GRAD_NORM_GPU_KERNEL_H_
