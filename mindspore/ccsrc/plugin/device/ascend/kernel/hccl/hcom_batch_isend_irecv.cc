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

#include "plugin/device/ascend/kernel/hccl/hcom_batch_isend_irecv.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace kernel {
bool HcomBatchISendIRecvKernel::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (!HcclKernel::Init(inputs, outputs)) {
    MS_LOG(ERROR) << "HcclKernel Init failed.";
    return false;
  }
  op_types_ = GetValue<std::vector<std::string>>(primitive_->GetAttr("op_types"));
  remote_ranks_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("remote_ranks"));
  return true;
}

int HcomBatchISendIRecvKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  auto shape_v = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("receive_shapes"));
  auto op_types = GetValue<std::vector<std::string>>(primitive_->GetAttr("op_types"));
  size_t output_index = 0;
  for (size_t i = 0; i < op_types.size(); ++i) {
    size_t size;
    auto type_ = op_types_[i];
    if (type_ == "isend") {
      size = SizeOf(outputs[i]->GetDeviceShapeVector());
    } else if (type_ == "irecv") {
      size = SizeOf(shape_v[output_index]);
      output_index++;
    } else {
      MS_LOG(EXCEPTION) << "HcclBatchISendIRecv only support 'isend' or 'irecv', but got "
                        << "'" << op_types_[i] << "'.";
    }
    output_size_list_.push_back(size);
  }
  return KRET_OK;
}

bool HcomBatchISendIRecvKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid hccl BatchISendIRecv input, output or data type size (" << inputs.size() << ", "
                  << outputs.size() << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }

  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto kItem = op_types_.size();
  HcclSendRecvItem info[kItem];
  HcclSendRecvType hccl_type;
  KernelTensor *tensor_to_hccl;
  size_t input_index = 0;
  for (size_t i = 0; i < kItem; i++) {
    auto type_ = op_types_[i];
    if (type_ == "isend") {
      hccl_type = HcclSendRecvType::HCCL_SEND;
      MS_EXCEPTION_IF_NULL(inputs[input_index]);
      tensor_to_hccl = inputs[input_index];
      input_index++;
    } else if (type_ == "irecv") {
      hccl_type = HcclSendRecvType::HCCL_RECV;
      MS_EXCEPTION_IF_NULL(outputs[i]);
      tensor_to_hccl = outputs[i];
    } else {
      MS_LOG(EXCEPTION) << "HcclBatchISendIRecv only support 'isend' or 'irecv', but got "
                        << "'" << op_types_[i] << "'.";
    }

    auto buf = tensor_to_hccl->device_ptr();
    MS_EXCEPTION_IF_NULL(buf);
    auto input_shape = tensor_to_hccl->GetDeviceShapeVector();
    auto numel = SizeOf(input_shape);
    auto hccl_dtype = HcomUtil::ConvertHcclType(tensor_to_hccl->dtype_id());
    auto rank = static_cast<uint32_t>(remote_ranks_[i]);

    info[i] = HcclSendRecvItem{hccl_type, buf, numel, hccl_dtype, rank};
  }

  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBatchISendIRecv(info, kItem, comm_, stream_ptr);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclBatchISendIRecv failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
