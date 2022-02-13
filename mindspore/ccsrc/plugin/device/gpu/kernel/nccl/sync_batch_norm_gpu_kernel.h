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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_

#include <dlfcn.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/nccl/nccl_gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "utils/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/sync_batch_norm_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S, typename G>
class SyncBatchNormGpuKernel : public NcclGpuKernelMod {
 public:
  SyncBatchNormGpuKernel() { ResetResource(); }
  ~SyncBatchNormGpuKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
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
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto root_rank = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrRootRank);
    if (root_rank) {
      root_ = static_cast<int>(GetValue<int64_t>(root_rank));
    }
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 5, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 5, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    auto input_shape_dims = input_shape.size();
    if (input_shape_dims != 4 && input_shape_dims != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input only should be 2 or 4, but got "
                        << input_shape_dims;
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
    SelectCollectiveHandle();
    // Get group size
    group_size_ = device::gpu::CollectiveInitializer::instance().GetGroupSize(group_name_);
    // // Get device rank ID in group
    group_rank_ = device::gpu::CollectiveInitializer::instance().local_rank_id();
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
    is_null_input_ = false;
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
    (void)AllGather(input_addr, output_addr, C_, nccl_dtype(GetTypeID(input_addr)), stream, group_name_);
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

  // NCCL
  string group_name_;
  int root_;
  cudaStream_t comm_stream_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_SYNC_BATCH_NORM_GPU_KERNEL_H_
