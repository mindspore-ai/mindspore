/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_all_broadcast.h"

#include <string>

#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace kernel {
bool HcomAllBroadCastKernel::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  bool ret = HcclKernel::Init(inputs, outputs);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Failed to init HcomAllBroadCastKernel";
  }
#ifdef ENABLE_INTERNAL_KERNELS
  std::string enable_lccl = device::ascend::EnableLcclEnv();
  if (enable_lccl == "on") {
    lccl_broadcast_func_ = DlsymFuncObj(Broadcast, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(lccl_broadcast_func_);
    use_lccl_ = true;
  }
#endif
  return true;
}

bool HcomAllBroadCastKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                    const std::vector<KernelTensor *> &, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcomAllBroadCast launch";
  if (inputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "BroadCast param is empty";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    auto lccl_result = lccl_broadcast_func_(lccl_ptr_, inputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0],
                                            root_id_, stream_ptr);
    if (lccl_result != Lcal::LCAL_SUCCESS) {
      MS_LOG(EXCEPTION) << "LCCL Broadcast failed.";
    }
  } else {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBroadcast(
      inputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0], root_id_, stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcomBroadcastOp : hcom_broadcast failed, return: " << hccl_result;
      return false;
    }
  }
#else
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBroadcast(
    inputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0], root_id_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomBroadcastOp : hcom_broadcast failed, return: " << hccl_result;
    return false;
  }
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
