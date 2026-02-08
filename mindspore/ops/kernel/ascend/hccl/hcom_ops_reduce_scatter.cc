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

#include "kernel/ascend/hccl/hcom_ops_reduce_scatter.h"
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool InnerCommReduceScatterKernel::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  return true;
}
uint64_t InnerCommReduceScatterKernel::GetReduceScatterVParam(const std::vector<KernelTensor *> &inputs,
                                                              uint64_t input_list_num) {
  const std::optional<int64_t> rank_size_opt;
  data_type_ = HcomUtil::ConvertHcclType(inputs[kIndex0]->dtype_id());
  std::vector<int64_t> input_split_sizes;
  uint64_t in_count;
  auto in_shape = inputs[kIndex0]->GetDeviceShapeVector();
  if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {in_shape}, 1, rank_size_opt, &in_count)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }
  if (!HcomUtil::GetHcomTypeSize(data_type_, &type_size_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }

  for (size_t i = 1; i < input_list_num - kIndex3; ++i) {
    uint64_t count;
    auto shape = inputs[i]->GetDeviceShapeVector();
    if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {shape}, 1, rank_size_opt, &count)) {
      MS_LOG(EXCEPTION) << "GetHcomCount fail!";
    }
    input_split_sizes.push_back(count);
    if (in_count != count) {
      same_shape_ = false;
      return in_count;
    }
  }

  params_.send_counts.clear();
  params_.sdispls.clear();
  send_size_byte_.clear();
  params_.recv_count = hccl_count_;
  uint64_t offset = 0;
  for (size_t i = 0; i < input_split_sizes.size(); i++) {
    auto count = LongToSize(input_split_sizes[i]);
    params_.send_counts.push_back(count);
    params_.sdispls.push_back(offset);
    send_size_byte_.push_back(count * type_size_);
    offset += count;
  }
  return params_.recv_count;
}
int InnerCommReduceScatterKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() < kInputNum5) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl DistCommReduceScatterKernel input size " << inputs.size();
  }
  input_size_ = inputs.size();
  rank_size_ = inputs[input_size_ - kIndex3]->GetValueWithCheck<int64_t>();
  if (rank_size_ != static_cast<int64_t>(input_size_ - kIndex4)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl DistCommReduceScatterKernel rank size " << rank_size_;
  }
  auto op_type = inputs[input_size_ - kIndex2]->GetValueWithCheck<std::string>();
  primitive_->AddAttr(kAttrRankSize, MakeValue(static_cast<int64_t>(rank_size_)));
  op_type_enum_ = HcomUtil::GetHcomReduceOpType(op_type);
  group_ = inputs[input_size_ - kIndex1]->GetValueWithCheck<std::string>();
  comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  const std::optional<int64_t> rank_size_opt;
  auto out_shape = outputs[kIndex0]->GetDeviceShapeVector();
  if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {out_shape}, 1, rank_size_opt, &hccl_count_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }
  auto output_numel = GetReduceScatterVParam(inputs, input_size_);
  ShapeVector shape;
  if (same_shape_) {
    shape = inputs[kIndex0]->GetDeviceShapeVector();
  } else {
    if (output_numel != 0) {
      shape.push_back(output_numel);
    }
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

bool InnerCommReduceScatterKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (same_shape_) {
    for (int r = 0; r < rank_size_; r++) {
      uint64_t offset = static_cast<uint64_t>(r * type_size_ * hccl_count_);
      auto data_ptr = inputs[r + kIndex1]->device_ptr();
      auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, static_cast<char *>(outputs[0]->device_ptr()) + offset,
                                    type_size_ * hccl_count_, data_ptr, type_size_ * hccl_count_,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
      if (cp_ret != EOK) {
        MS_LOG(ERROR) << "aclrtMemcpy failed.";
        return false;
      }
    }
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatter(
      outputs[0]->device_ptr(), inputs[0]->device_ptr(), hccl_count_, data_type_, op_type_enum_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcclReduceScatter failed, ret:" << hccl_result;
      return false;
    }
  } else {
    uint64_t offset = 0;
    for (int r = 0; r < rank_size_; r++) {
      auto data_ptr = inputs[r + kIndex1]->device_ptr();
      if (send_size_byte_[r] == 0) {
        continue;
      }
      auto cp_ret =
        CALL_ASCEND_API(aclrtMemcpyAsync, static_cast<char *>(outputs[0]->device_ptr()) + offset, send_size_byte_[r],
                        data_ptr, send_size_byte_[r], ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
      if (cp_ret != EOK) {
        MS_LOG(ERROR) << "aclrtMemcpy failed.";
        return false;
      }
      offset += send_size_byte_[r];
    }
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatterV(
      outputs[0]->device_ptr(), inputs[0]->device_ptr(), params_, data_type_, op_type_enum_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(EXCEPTION) << "HcclReduceScatterV failed, ret:" << hccl_result;
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
