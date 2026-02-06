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

#include "kernel/ascend/hccl/hcom_ops_reduce_scatter_tensor.h"
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_manager.h"

namespace mindspore {
namespace kernel {
bool InnerCommReduceScatterTensorKernel::Init(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  inplace_num_ = 0;
  return true;
}

int InnerCommReduceScatterTensorKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != inplace_num_ + kInputNum4) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl InnerCommReduceScatterTensorKernel input size " << inputs.size();
  }
  rank_size_ = inputs[inplace_num_ + kIndex1]->GetValueWithCheck<int64_t>();
  auto op_type = inputs[inplace_num_ + kIndex2]->GetValueWithCheck<std::string>();
  primitive_->AddAttr(kAttrRankSize, MakeValue(static_cast<int64_t>(rank_size_)));
  op_type_enum_ = HcomUtil::GetHcomReduceOpType(op_type);
  group_ = inputs[inplace_num_ + kIndex3]->GetValueWithCheck<std::string>();

  data_type_ = HcomUtil::ConvertHcclType(inputs[inplace_num_]->dtype_id());
  const std::optional<int64_t> rank_size_opt;
  auto input_shape = inputs[inplace_num_]->GetDeviceShapeVector();
  auto output_shape = outputs[kIndex0]->GetDeviceShapeVector();
  if (!HcomUtil::GetHcomCount(primitive_, {data_type_}, {input_shape}, 1, rank_size_opt, &hccl_count_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail!";
  }
  comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  output_size_list_.clear();
  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(data_type_, output_shape, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  if (!outputs.empty()) {
    output_size_list_.push_back(size);
  }

  return KRET_OK;
}

bool InnerCommReduceScatterTensorKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[inplace_num_]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto hccl_result =
    hccl::HcclAdapter::GetInstance().HcclReduceScatter(inputs[inplace_num_]->device_ptr(), outputs[0]->device_ptr(),
                                                       hccl_count_, data_type_, op_type_enum_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclReduceScatter failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool DistCommReduceScatterTensorKernel::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  inplace_num_ = 1;
  return true;
}

}  // namespace kernel
}  // namespace mindspore
