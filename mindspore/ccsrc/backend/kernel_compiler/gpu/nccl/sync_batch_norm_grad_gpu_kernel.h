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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GRAD_GPU_KERNEL_H_

#include <dlfcn.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sync_batch_norm_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S, typename G>
class SyncBatchNormGradGpuKernel : public NcclGpuKernel {
 public:
  SyncBatchNormGradGpuKernel() { ResetResource(); }
  ~SyncBatchNormGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *x_input = GetDeviceAddress<T>(inputs, 1);
    S *scale = GetDeviceAddress<S>(inputs, 2);
    G *saved_mean = GetDeviceAddress<G>(inputs, 3);
    G *saved_variance = GetDeviceAddress<G>(inputs, 4);
    float *dy_sum_local = GetDeviceAddress<float>(workspace, 0);
    float *dot_p_local = GetDeviceAddress<float>(workspace, 1);
    float *dy_sum_red = GetDeviceAddress<float>(workspace, 2);
    float *dot_p_red = GetDeviceAddress<float>(workspace, 3);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    S *dscale = GetDeviceAddress<S>(outputs, 1);
    S *dbias = GetDeviceAddress<S>(outputs, 2);
    // aggregate interim values on each device locally
    CalSyncBatchNormGradPre(N_, C_, H_, W_, x_input, dy, saved_mean, saved_variance, dy_sum_local, dot_p_local,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    // reduce values across devices
    LaunchAllReduce(dy_sum_local, dy_sum_red, stream_ptr);
    LaunchAllReduce(dot_p_local, dot_p_red, stream_ptr);
    // Aggregate and compute output
    CalSyncBatchNormGradPost(N_, C_, H_, W_, x_input, dy, dx, saved_mean, saved_variance, dy_sum_red, dot_p_red, scale,
                             dscale, dbias, epsilon_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto root_rank = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrRootRank);
    if (root_rank) {
      root_ = static_cast<int>(GetValue<int64_t>(root_rank));
    }
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SyncBatchNormGrad needs 5 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 3) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but SyncBatchNormGrad needs 5 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (CHECK_NULL_INPUT(input_shape)) {
      MS_LOG(WARNING) << "SyncBatchNormGrad input is null";
      InitSizeLists();
      return true;
    }
    auto input_shape_dims = input_shape.size();
    if (input_shape_dims != 4 && input_shape_dims != 2) {
      MS_LOG(EXCEPTION) << "Tensor shape is " << input_shape.size()
                        << ", SyncBatchNormGpuGrad input should be 2D or 4D";
    }
    input_size_ = 1;
    for (auto dim : input_shape) {
      input_size_ *= dim;
    }
    output_size_ = input_size_;
    output_size_ = output_size_ * sizeof(T);
    input_size_ = input_size_ * sizeof(T);
    param_count_ = input_shape[1];
    param_size_S_ = param_count_ * sizeof(S);
    param_size_G_ = param_count_ * sizeof(G);
    N_ = input_shape[0];
    C_ = input_shape[1];
    if (input_shape_dims == 2) {  // N,C,1,1 transform input
      H_ = 1;
      W_ = 1;
    } else {
      H_ = input_shape[2];
      W_ = input_shape[3];
    }
    workspace_size_ = C_;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    // MULTIDEVICE SPECIFICS
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    MS_LOG(INFO) << AnfAlgo::GetCNodeName(kernel_node) << " for group " << group_name_;
    auto comm_stream_attr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stream_id");
    if (comm_stream_attr) {
      comm_stream_ = reinterpret_cast<cudaStream_t>(GetValue<uintptr_t>(comm_stream_attr));
      MS_EXCEPTION_IF_NULL(comm_stream_);
    }
    collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
    MS_EXCEPTION_IF_NULL(collective_handle_);
    // Get group size
    auto get_group_size_funcptr =
      reinterpret_cast<GetGroupRanks>(dlsym(const_cast<void *>(collective_handle_), "GetGroupRanks"));
    MS_EXCEPTION_IF_NULL(get_group_size_funcptr);
    std::vector<int> group_ranks = (*get_group_size_funcptr)(group_name_);
    device_count_ = group_ranks.size();
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    epsilon_ = 10e-5;  // default
    param_size_S_ = 0;
    param_size_G_ = 0;
    param_count_ = 0;
    N_ = 0;
    C_ = 0;
    H_ = 0;
    W_ = 0;
    root_ = 0;
    collective_handle_ = nullptr;
    comm_stream_ = nullptr;
    nccl_reduce_type_ = ncclSum;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);                          // dy
    input_size_list_.push_back(input_size_);                          // x
    input_size_list_.push_back(param_size_S_);                        // scale
    input_size_list_.push_back(param_size_G_);                        // saved_mean
    input_size_list_.push_back(param_size_G_);                        // saved_variance
    output_size_list_.push_back(output_size_);                        // dx
    output_size_list_.push_back(param_size_S_);                       // dscale
    output_size_list_.push_back(param_size_S_);                       // dbias
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // sum_dy
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // sum_dy_xmu
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // reduced sum_dy
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // reduced sum_dy_xmu
  }

 private:
  template <typename reduce_type>
  void LaunchAllReduce(reduce_type *input_addr, reduce_type *output_addr, void *stream_ptr) {
    cudaStream_t stream = comm_stream_ ? comm_stream_ : reinterpret_cast<cudaStream_t>(stream_ptr);
    auto all_reduce_funcptr = reinterpret_cast<AllReduce>(dlsym(const_cast<void *>(collective_handle_), "AllReduce"));
    MS_EXCEPTION_IF_NULL(all_reduce_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(kernel_node_,
                               (*all_reduce_funcptr)(input_addr, output_addr, C_, nccl_dtype(kNumberTypeFloat32),
                                                     nccl_reduce_type_, stream, group_name_),
                               "ncclAllReduce - SyncBatchNormGrad - CUDA failed");
  }

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  float epsilon_;
  size_t param_size_S_;
  size_t param_size_G_;
  size_t param_count_;
  size_t N_;
  size_t C_;
  size_t H_;
  size_t W_;
  size_t device_count_;
  ncclRedOp_t nccl_reduce_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  // NCCL
  string group_name_;
  int root_;
  const void *collective_handle_;
  cudaStream_t comm_stream_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GRAD_GPU_KERNEL_H_
