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

#include "plugin/device/ascend/kernel/hccl/hcom_all_gather.h"

#include <string>

#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace kernel {
bool HcomAllGatherKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  bool ret = HcclKernel::Init(inputs, outputs);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Failed to init HcomAllGatherKernel";
  }
#ifdef ENABLE_INTERNAL_KERNELS
  std::string enable_lccl = device::ascend::EnableLcclEnv();
  if (enable_lccl == "on") {
    lccl_all_gather_func_ = DlsymFuncObj(AllGather, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(lccl_all_gather_func_);
    use_lccl_ = true;
  }
#endif
  return true;
}

bool HcomAllGatherKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcomAllGather launch";
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid AllGather input, output or data type size(" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    auto lccl_result = lccl_all_gather_func_(lccl_ptr_, inputs[0]->device_ptr(), outputs[0]->device_ptr(), hccl_count_,
                                             hccl_data_type_list_[0], stream_ptr);
    if (lccl_result != Lcal::LCAL_SUCCESS) {
      MS_LOG(EXCEPTION) << "LCCL AllGather failed.";
    }
  } else {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(
      inputs[0]->device_ptr(), outputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0], stream_ptr, comm_);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcclAllGather failed, ret:" << hccl_result;
      return false;
    }
  }
#else
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(
    inputs[0]->device_ptr(), outputs[0]->device_ptr(), hccl_count_, hccl_data_type_list_[0], stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllGather failed, ret:" << hccl_result;
    return false;
  }
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
