/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_

#include <dlfcn.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sync_batch_norm_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S, typename G>
class SyncBatchNormGpuKernel : public NcclGpuKernel {
 public:
  SyncBatchNormGpuKernel() { ResetResource(); }
  ~SyncBatchNormGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *x = GetDeviceAddress<T>(inputs, 0);
    S *scale = GetDeviceAddress<S>(inputs, 1);
    S *bias = GetDeviceAddress<S>(inputs, 2);
    G *running_mean_input = GetDeviceAddress<G>(inputs, 3);
    G *running_variance_input = GetDeviceAddress<G>(inputs, 4);

    float *means_local = GetDeviceAddress<float>(workspace, 0);  // per device
    float *invstds_local = GetDeviceAddress<float>(workspace, 1);
    int *counts_local = GetDeviceAddress<int>(workspace, 2);
    int *counts_global = GetDeviceAddress<int>(workspace, 3);  // gathered values from all devices
    float *means_global = GetDeviceAddress<float>(workspace, 4);
    float *invstds_global = GetDeviceAddress<float>(workspace, 5);

    T *y = GetDeviceAddress<T>(outputs, 0);
    S *output_scale = GetDeviceAddress<S>(outputs, 1);
    S *output_bias = GetDeviceAddress<S>(outputs, 2);
    T *output_running_mean = GetDeviceAddress<T>(outputs, 3);
    T *output_running_variance = GetDeviceAddress<T>(outputs, 4);

    // aggregate means and invstd on each device locally
    CalSyncBatchNormPre(N_, C_, H_, W_, x, counts_local, means_local, invstds_local, epsilon_,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    // gather values from all devices together
    LaunchAllGather(means_local, means_global, stream_ptr);
    LaunchAllGather(invstds_local, invstds_global, stream_ptr);
    LaunchAllGather(counts_local, counts_global, stream_ptr);
    // reducing gathered values on each device and deal with running means and variance
    CalSyncBatchNormGather(N_, C_, H_, W_, counts_global, means_global, invstds_global, counts_local, means_local,
                           invstds_local, output_running_mean, output_running_variance, running_mean_input,
                           running_variance_input, epsilon_, momentum_, group_rank_, group_size_,
                           reinterpret_cast<cudaStream_t>(stream_ptr));
    CalSyncBatchNormPost(N_, C_, H_, W_, x, y, means_local, invstds_local, scale, bias, output_scale, output_bias,
                         epsilon_, reinterpret_cast<cudaStream_t>(stream_ptr));
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
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SyncBatchNorm needs 5 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 5) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but SyncBatchNorm needs 5 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (CHECK_NULL_INPUT(input_shape)) {
      MS_LOG(WARNING) << "SyncBatchNorm input is null";
      InitSizeLists();
      return true;
    }
    auto input_shape_dims = input_shape.size();
    if (input_shape_dims != 4 && input_shape_dims != 2) {
      MS_LOG(EXCEPTION) << "Tensor shape is " << input_shape.size()
                        << ", SyncBatchNormGpuKernel input should be 2D or 4D";
    }
    input_size_ = 1;
    for (auto dim : input_shape) {
      input_size_ *= dim;
    }
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    momentum_ = GetAttr<float>(kernel_node, "momentum");
    output_size_ = input_size_;
    output_size_ = output_size_ * sizeof(T);
    input_size_ = input_size_ * sizeof(T);
    param_count_ = input_shape[1];             // C is number of features
    param_size_S_ = param_count_ * sizeof(S);  // will be second/third template
    param_size_G_input_ = param_count_ * sizeof(G);
    param_size_G_output_ = param_count_ * sizeof(T);
    workspace_size_ = param_count_;  // specific size computed in InitSizeLists()
    N_ = input_shape[0];
    C_ = input_shape[1];
    if (input_shape_dims == 2) {
      // NC -> N,C,1,1 transform input dims
      H_ = 1;
      W_ = 1;
    } else {
      H_ = input_shape[2];
      W_ = input_shape[3];
    }
    // MULTI DEVICE SPECIFICS
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
    group_size_ = group_ranks.size();
    // // Get device rank ID in group
    using GetLocalRankId = device::gpu::GetLocalRankId;
    auto get_local_rank_funcptr =
      reinterpret_cast<GetLocalRankId>(dlsym(const_cast<void *>(collective_handle_), "local_rank_id"));
    MS_EXCEPTION_IF_NULL(get_local_rank_funcptr);
    group_rank_ = IntToUint((*get_local_rank_funcptr)());
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    momentum_ = 0;
    epsilon_ = 10e-5;
    param_size_S_ = 0;
    param_size_G_input_ = 0;
    param_size_G_output_ = 0;
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
    input_size_list_.push_back(input_size_);            // input x
    input_size_list_.push_back(param_size_S_);          // scale
    input_size_list_.push_back(param_size_S_);          // bias
    input_size_list_.push_back(param_size_G_input_);    // running mean
    input_size_list_.push_back(param_size_G_input_);    // running variance
    output_size_list_.push_back(output_size_);          // output
    output_size_list_.push_back(param_size_S_);         // save scale
    output_size_list_.push_back(param_size_S_);         // reserve space
    output_size_list_.push_back(param_size_G_output_);  // save mean
    output_size_list_.push_back(param_size_G_output_);  // save variance
    // local mean/variance data - per device
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // mean_local
    workspace_size_list_.push_back(workspace_size_ * sizeof(float));  // invstd_local
    workspace_size_list_.push_back(workspace_size_ * sizeof(int));    // count_local
    // global mean/variance data - for all devices
    workspace_size_list_.push_back(workspace_size_ * sizeof(int) * group_size_);    // gathered mean
    workspace_size_list_.push_back(workspace_size_ * sizeof(float) * group_size_);  // gathered invstd
    workspace_size_list_.push_back(workspace_size_ * sizeof(float) * group_size_);  // gathered count
  }

 private:
  // GetTypeID functions return the correct typeID for input template
  // Allow for a single templated LaunchAllGather function
  mindspore::TypeId GetTypeID(float *input) { return kNumberTypeFloat32; }
  mindspore::TypeId GetTypeID(int *input) { return kNumberTypeInt32; }
  template <typename gather_type>
  void LaunchAllGather(gather_type *input_addr, gather_type *output_addr, void *stream_ptr) {
    cudaStream_t stream = comm_stream_ ? comm_stream_ : reinterpret_cast<cudaStream_t>(stream_ptr);
    auto all_gather_funcptr = reinterpret_cast<AllGather>(dlsym(const_cast<void *>(collective_handle_), "AllGather"));
    MS_EXCEPTION_IF_NULL(all_gather_funcptr);
    CHECK_NCCL_RET_WITH_EXCEPT(
      kernel_node_,
      (*all_gather_funcptr)(input_addr, output_addr, C_, nccl_dtype(GetTypeID(input_addr)), stream, group_name_),
      "ncclAllGather failed");
  }

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  float momentum_;
  float epsilon_;
  size_t param_size_S_;
  size_t param_size_G_input_;
  size_t param_size_G_output_;
  size_t param_count_;
  size_t N_;
  size_t C_;
  size_t H_;
  size_t W_;
  size_t group_size_;
  size_t group_rank_;
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
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_
