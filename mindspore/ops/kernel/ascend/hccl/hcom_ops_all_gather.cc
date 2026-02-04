/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/hccl/hcom_ops_all_gather.h"
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool DistCommAllGatherKernel::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  return true;
}

uint64_t DistCommAllGatherKernel::GetAllGatherVParam(const std::vector<KernelTensor *> &inputs,
                                                     uint64_t input_list_num) {
  const std::optional<int64_t> rank_size_opt;
  data_type_ = HcomUtil::ConvertHcclType(inputs[input_size_ - kIndex3]->dtype_id());
  std::vector<int64_t> output_split_sizes;
  auto in_shape = inputs[input_size_ - kIndex3]->GetDeviceShapeVector();
  if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {in_shape}, 1, rank_size_opt, &hccl_count_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }
  if (!HcomUtil::GetHcomTypeSize(data_type_, &type_size_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }

  for (size_t i = 0; i < input_list_num - kIndex3; ++i) {
    uint64_t count;
    auto shape = inputs[i]->GetDeviceShapeVector();
    if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {shape}, 1, rank_size_opt, &count)) {
      MS_LOG(EXCEPTION) << "GetHcomCount fail!";
    }
    output_split_sizes.push_back(count);
    if (hccl_count_ != count) {
      same_shape_ = false;
      return hccl_count_;
    }
  }

  params_.send_count = hccl_count_;
  params_.recv_counts.clear();
  params_.rdispls.clear();
  recv_size_byte_.clear();
  uint64_t offset = 0;
  for (size_t i = 0; i < output_split_sizes.size(); i++) {
    auto count = LongToSize(output_split_sizes[i]);
    params_.recv_counts.push_back(count);
    params_.rdispls.push_back(offset);
    recv_size_byte_.push_back(count * type_size_);
    offset += count;
  }

  return offset;
}

int DistCommAllGatherKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() < kInputNum4) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl DistCommAllGatherKernel input size " << inputs.size();
  }
  input_size_ = inputs.size();
  rank_size_ = inputs[input_size_ - kIndex2]->GetValueWithCheck<int64_t>();
  if (rank_size_ != static_cast<int64_t>(input_size_ - kIndex3)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl DistCommAllGatherKernel rank size " << rank_size_;
  }
  group_ = inputs[input_size_ - kIndex1]->GetValueWithCheck<std::string>();
  uint64_t output_numel = GetAllGatherVParam(inputs, input_size_);

  ShapeVector shape;
  if (same_shape_) {
    shape = inputs[input_size_ - kIndex3]->GetDeviceShapeVector();
  } else {
    if (output_numel != 0) {
      shape.push_back(output_numel);
    }
  }
  comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  output_size_list_.clear();
  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(data_type_, shape, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  if (!outputs.empty()) {
    output_size_list_.push_back(size);
  }

  return KRET_OK;
}

bool DistCommAllGatherKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[input_size_ - kIndex3]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (same_shape_) {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(inputs[input_size_ - kIndex3]->device_ptr(),
                                                                      outputs[kIndex0]->device_ptr(), hccl_count_,
                                                                      data_type_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcclAllGather failed, ret:" << hccl_result;
      return false;
    }

    for (int r = 0; r < rank_size_; r++) {
      uint64_t offset = static_cast<uint64_t>(r * type_size_ * hccl_count_);
      auto data_ptr = inputs[r]->device_ptr();
      auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, type_size_ * hccl_count_,
                                    static_cast<char *>(outputs[kIndex0]->device_ptr()) + offset,
                                    type_size_ * hccl_count_, ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
      if (cp_ret != EOK) {
        MS_LOG(ERROR) << "aclrtMemcpy failed.";
        return false;
      }
    }
  } else {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGatherV(inputs[input_size_ - kIndex3]->device_ptr(),
                                                                       outputs[kIndex0]->device_ptr(), params_,
                                                                       data_type_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcclAllGatherV failed, ret:" << hccl_result;
      return false;
    }
    uint64_t offset = 0;
    for (int r = 0; r < rank_size_; r++) {
      if (recv_size_byte_[r] == 0) {
        continue;
      }
      auto data_ptr = inputs[r]->device_ptr();
      auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, recv_size_byte_[r],
                                    static_cast<char *>(outputs[kIndex0]->device_ptr()) + offset, recv_size_byte_[r],
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
      if (cp_ret != EOK) {
        MS_LOG(ERROR) << "HcclAllGather aclrtMemcpy failed.";
        return false;
      }
      offset += recv_size_byte_[r];
    }
  }
  return true;
}

}  // namespace kernel
}  // namespace mindspore
