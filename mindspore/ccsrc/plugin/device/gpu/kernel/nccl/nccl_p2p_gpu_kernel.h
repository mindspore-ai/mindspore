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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_P2P_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_P2P_GPU_KERNEL_H_

#include <dlfcn.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace kernel {
enum NcclKernelType { NCCL_ALLTOALLV = 0, NCCL_NEIGHBOREXCHANGE = 1, NCCL_INVALID_TYPE = 255 };
const std::map<std::string, NcclKernelType> kNcclTypeMap = {{"AllToAllv", NCCL_ALLTOALLV},
                                                            {"NeighborExchange", NCCL_NEIGHBOREXCHANGE}};

template <typename T, typename I>
class NcclP2PGpuKernel : public NcclGpuKernelMod {
 public:
  NcclP2PGpuKernel() { ResetResource(); }
  ~NcclP2PGpuKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    switch (nccl_kernel_type_) {
      case NCCL_ALLTOALLV: {
        LaunchAllToAllv(inputs, outputs, stream_ptr);
        break;
      }
      case NCCL_NEIGHBOREXCHANGE: {
        LaunchAllToAllv(inputs, outputs, stream_ptr);
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: AllToAllv, NeighborExchange "
                          << "currently, but got " << nccl_kernel_type_;
      }
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    InferCommType(kernel_node);

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (input_num > 0) {
      input_nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    }
    if (output_num > 0) {
      output_nccl_data_type_ = nccl_dtype(AnfAlgo::GetOutputDeviceDataType(kernel_node, 0));
    }
    for (size_t i = 0; i < input_num; ++i) {
      auto shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, i);
      is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      size_t size = sizeof(T);
      for (size_t j = 0; j < shape.size(); j++) {
        size *= LongToSizeClipNeg(shape[j]);
      }
      input_size_list_.push_back(size);
      // Framework memory allocation ensures memory alignment.
      input_size_ += device::gpu::GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    }
    for (size_t i = 0; i < output_num; ++i) {
      auto shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, i);
      is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "output");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      size_t size = sizeof(I);
      for (size_t j = 0; j < shape.size(); j++) {
        size *= LongToSizeClipNeg(shape[j]);
      }
      output_size_list_.push_back(size);
      // Framework memory allocation ensures memory alignment.
      output_size_ += device::gpu::GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    }

    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    MS_LOG(INFO) << common::AnfAlgo::GetCNodeName(kernel_node) << " for group " << group_name_;

    // Used by AlltoAllv
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto send_rank_ids_attr = prim->GetAttr(kAttrSendRankIds);
    auto recv_rank_ids_attr = prim->GetAttr(kAttrRecvRankIds);
    if (send_rank_ids_attr && recv_rank_ids_attr) {
      send_rank_ids_ = GetValue<std::vector<int64_t>>(send_rank_ids_attr);
      recv_rank_ids_ = GetValue<std::vector<int64_t>>(recv_rank_ids_attr);
    }

    auto need_drop_input_attr = prim->GetAttr(kAttrNeedDropInput);
    if (need_drop_input_attr) {
      need_drop_input_ = GetValue<bool>(need_drop_input_attr);
    }

    SelectCollectiveHandle();
    return true;
  }

  void ResetResource() noexcept override {
    nccl_kernel_type_ = NCCL_INVALID_TYPE;
    input_size_ = 0;
    output_size_ = 0;
    root_ = 0;
    is_null_input_ = false;
    collective_handle_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    need_drop_input_ = false;
  }

 protected:
  void InitSizeLists() override { return; }

 private:
  void LaunchAllToAllv(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                       void *stream_ptr) {
    T *input_addr = nullptr;
    I *output_addr = nullptr;

    // send_rank_id and recv rank_id size needs to be equal to input_list size, unless there is a depend input in
    // the input_list.
    if (send_rank_ids_.size() != input_size_list_.size() && !need_drop_input_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", trying to use AlltoAllv, the size of send_rank_ids vector "
                        << "should be " << input_size_list_.size() << ", but got " << send_rank_ids_.size();
    }
    if (recv_rank_ids_.size() != output_size_list_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", trying to use AlltoAllv, the size of recv_rank_ids vector "
                        << "should be " << output_size_list_.size() << ", but got " << recv_rank_ids_.size();
    }

    // This implementation refers to NVIDIA NCCL 2.11 doc.
    (void)GroupStart();
    if (!need_drop_input_) {
      for (size_t i = 0; i < input_size_list_.size(); ++i) {
        input_addr = GetDeviceAddress<T>(inputs, i);
        (void)Send(input_addr, input_size_list_[i] / sizeof(T), input_nccl_data_type_, send_rank_ids_[i],
                   reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);
      }
    }
    for (size_t i = 0; i < output_size_list_.size(); ++i) {
      output_addr = GetDeviceAddress<I>(outputs, i);
      (void)Recv(output_addr, output_size_list_[i] / sizeof(I), output_nccl_data_type_, recv_rank_ids_[i],
                 reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);
    }
    (void)GroupEnd();
  }

  void InferCommType(const CNodePtr &kernel_node) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kNcclTypeMap.find(kernel_name);
    if (iter == kNcclTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: AllToAllv, NeighborExchange "
                        << "currently, but got " << kernel_name;
    } else {
      nccl_kernel_type_ = iter->second;
    }

    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);

    auto root_rank = prim->GetAttr(kAttrRootRank);
    if (root_rank) {
      root_ = static_cast<int>(GetValue<int64_t>(root_rank));
    }
    return;
  }

  NcclKernelType nccl_kernel_type_;
  size_t input_size_;
  size_t output_size_;
  int root_;
  bool is_null_input_;
  ncclDataType_t output_nccl_data_type_;
  ncclDataType_t input_nccl_data_type_;
  std::vector<int64_t> send_rank_ids_;
  std::vector<int64_t> recv_rank_ids_;
  bool need_drop_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_P2P_GPU_KERNEL_H_
