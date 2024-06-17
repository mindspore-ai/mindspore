/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_all_to_all_v.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace kernel {
bool HcomAlltoAllVKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!HcclKernel::Init(inputs, outputs)) {
    MS_LOG(ERROR) << "HcclKernel Init failed.";
    return false;
  }
  auto send_numel_list = GetValue<std::vector<int64_t>>(primitive_->GetAttr("send_numel_list"));
  auto recv_numel_list = GetValue<std::vector<int64_t>>(primitive_->GetAttr("recv_numel_list"));
  uint64_t offset = 0;
  for (size_t i = 0; i < send_numel_list.size(); i++) {
    auto count = static_cast<uint64_t>(send_numel_list[i]);
    params_.sendcounts.push_back(count);
    params_.sdispls.push_back(offset);
    offset += count;
  }
  offset = 0;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    auto count = static_cast<uint64_t>(recv_numel_list[i]);
    params_.recvcounts.push_back(count);
    params_.rdispls.push_back(offset);
    offset += count;
  }
  return true;
}

int HcomAlltoAllVKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  auto recv_numel_list = GetValue<std::vector<int64_t>>(primitive_->GetAttr("recv_numel_list"));
  int64_t output_numel = 0;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    output_numel += recv_numel_list[i];
  }
  if (output_numel == 0) {
    output_size_list_.push_back(SizeOf(ShapeVector{}));
  } else {
    output_size_list_.push_back(SizeOf(ShapeVector{output_numel}));
  }
  return KRET_OK;
}

bool HcomAlltoAllVKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid hccl AlltoAllV input, output or data type size (" << inputs.size() << ", "
                  << outputs.size() << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }

  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto send_tensor = inputs[0];
  auto recv_tensor = outputs[0];
  MS_EXCEPTION_IF_NULL(send_tensor);
  MS_EXCEPTION_IF_NULL(recv_tensor);

  auto send_buf = send_tensor->device_ptr();
  auto recv_buf = recv_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(send_buf);
  MS_EXCEPTION_IF_NULL(recv_buf);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllToAllv(send_buf, recv_buf, params_,
                                                                    hccl_data_type_list_[0], stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllToAllv failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
