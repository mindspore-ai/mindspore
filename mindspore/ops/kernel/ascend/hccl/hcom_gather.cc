/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/hccl/hcom_gather.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "include/cluster/init.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool HcomGatherKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!HcclKernel::Init(inputs, outputs)) {
    MS_LOG(ERROR) << "HcclKernel Init failed.";
    return false;
  }
  rank_id_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("rank_id")));
  dest_rank_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("dest_rank")));
  rank_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("rank_size")));
  return true;
}

int HcomGatherKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto res = HcclKernel::Resize(inputs, outputs); res != KRET_OK) {
    MS_LOG(ERROR) << "HcclKernel Resize failed.";
    return res;
  }
  auto input_shape = inputs[0]->GetDeviceShapeVector();
  hccl_count_ = SizeOf(input_shape);
  return KRET_OK;
}

bool HcomGatherKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcomGather launch";
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid Gather input, output or data type size(" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto data_type = hccl_data_type_list_[0];
  if (data_type == HCCL_DATA_TYPE_FP16) {
    return LaunchKernel<float16>(inputs, outputs, stream_ptr);
  } else if (data_type == HCCL_DATA_TYPE_FP32) {
    return LaunchKernel<float>(inputs, outputs, stream_ptr);
  } else if (data_type == HCCL_DATA_TYPE_INT8) {
    return LaunchKernel<int8_t>(inputs, outputs, stream_ptr);
  } else if (data_type == HCCL_DATA_TYPE_INT32) {
    return LaunchKernel<int32_t>(inputs, outputs, stream_ptr);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type: " << data_type;
  }
  return true;
}

template <typename T>
bool HcomGatherKernel::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid hccl Gather input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto input_size = inputs[0]->size();
  T *output_device_ptr = static_cast<T *>(outputs[0]->device_ptr());
  if (rank_id_ == dest_rank_) {
    for (int r = 0; r < rank_size_; r++) {
      int offset = r * hccl_count_;
      if (r == dest_rank_) {
        auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, output_device_ptr + offset, input_size, inputs[0]->device_ptr(),
                                      input_size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
        if (cp_ret != EOK) {
          MS_LOG(ERROR) << "aclrtMemcpy failed.";
          return false;
        }
      } else {
        if (NeedReGetHcom()) {
          MS_LOG(WARNING) << "Hccl inner name had changed, need re-get hcom";
          comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
        }
        auto hccl_result = hccl::HcclAdapter::GetInstance().HcclRecv(output_device_ptr + offset, hccl_count_,
                                                                     hccl_data_type_list_[0], r, stream_ptr, comm_);
        if (hccl_result != HCCL_SUCCESS) {
          MS_LOG(ERROR) << "HcclRecv failed, ret:" << hccl_result;
          return false;
        }
      }
    }
  } else {
    if (NeedReGetHcom()) {
      MS_LOG(WARNING) << "Hccl inner name had changed, need re-get hcom";
      comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
    }
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclSend(
      inputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0], dest_rank_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcclSend failed, ret:" << hccl_result;
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
