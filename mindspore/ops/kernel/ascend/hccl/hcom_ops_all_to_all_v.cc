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

#include "kernel/ascend/hccl/hcom_ops_all_to_all_v.h"
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_manager.h"

namespace mindspore {
namespace kernel {
bool InnerCommAllToAllVSingleKernel::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  inplace_num_ = 0;
  return true;
}

void InnerCommAllToAllVSingleKernel::GetAllToAllVParam(const std::vector<int64_t> &send_numel_list,
                                                       const std::vector<int64_t> &recv_numel_list) {
  params_.sendcounts.clear();
  params_.sdispls.clear();
  params_.recvcounts.clear();
  params_.rdispls.clear();
  uint64_t offset = 0;
  for (size_t i = 0; i < send_numel_list.size(); i++) {
    auto count = LongToSize(send_numel_list[i]);
    params_.sendcounts.push_back(count);
    params_.sdispls.push_back(offset);
    offset += count;
  }
  offset = 0;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    auto count = LongToSize(recv_numel_list[i]);
    params_.recvcounts.push_back(count);
    params_.rdispls.push_back(offset);
    offset += count;
  }
}

int InnerCommAllToAllVSingleKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  std::vector<int64_t> send_numel_list;
  std::vector<int64_t> recv_numel_list;
  if (inputs.size() == kIndex6 + inplace_num_) {
    send_numel_list = inputs[kIndex2 + inplace_num_]->GetValueWithCheck<std::vector<int64_t>>();
    recv_numel_list = inputs[kIndex3 + inplace_num_]->GetValueWithCheck<std::vector<int64_t>>();
    GetAllToAllVParam(send_numel_list, recv_numel_list);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl AlltoAllV input size " << inputs.size();
  }
  data_type_ = HcomUtil::ConvertHcclType(inputs[kIndex0 + inplace_num_]->dtype_id());
  rank_size_ = inputs[kIndex4 + inplace_num_]->GetValueWithCheck<int64_t>();
  group_ = inputs[kIndex1 + inplace_num_]->GetValueWithCheck<std::string>();
  comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  int64_t output_numel = 0;
  ShapeVector shape;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    output_numel += recv_numel_list[i];
  }
  if (output_numel != 0) {
    shape.push_back(output_numel);
  }

  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(data_type_, shape, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  if (!outputs.empty()) {
    output_size_list_.push_back(size);
  }
  return KRET_OK;
}

bool InnerCommAllToAllVSingleKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto send_tensor = inputs[kIndex0 + inplace_num_];
  MS_EXCEPTION_IF_NULL(send_tensor);
  auto send_buf = send_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(send_buf);
  void *recv_buf = nullptr;
  if (!outputs.empty()) {  // may be empty output when AlltoAllV is from NeighborExchangeV2
    auto recv_tensor = outputs[0];
    MS_EXCEPTION_IF_NULL(recv_tensor);
    recv_buf = recv_tensor->device_ptr();
    MS_EXCEPTION_IF_NULL(recv_buf);
  }
  auto hccl_result =
    hccl::HcclAdapter::GetInstance().HcclAlltoAllV(send_buf, recv_buf, params_, data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAlltoAllV failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool DistCommAllToAllVSingleKernel::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  inplace_num_ = 1;
  return true;
}

}  // namespace kernel
}  // namespace mindspore
