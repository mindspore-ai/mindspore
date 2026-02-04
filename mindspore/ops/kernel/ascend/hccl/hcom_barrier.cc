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

#include "kernel/ascend/hccl/hcom_barrier.h"

#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
// We use 4 bytes data for AllReduce operator as barrier.
constexpr size_t kBarrierDataSize = 4;
bool HcomBarrierKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!HcclKernel::Init(inputs, outputs)) {
    MS_LOG(ERROR) << "Call HcclKernel::Init failed.";
    return false;
  }

#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    MS_LOG(INFO) << "Use LCCL AllReduce as Barrier.";
    // If using LCCL, use AllReduce to attain barrier semantics.
    lccl_all_reduce_func_ = DlsymFuncObj(AllReduce, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(lccl_all_reduce_func_);

    auto context_ptr = mindspore::MsContext::GetInstance();
    uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::GetDeviceTypeByName(device_name), device_id});
    lccl_barrier_data_ = device_context->device_res_manager_->AllocateMemory(kBarrierDataSize);
    MS_EXCEPTION_IF_NULL(lccl_barrier_data_);
    // Set buffer value to all 0.
    auto ret = aclrtMemset(lccl_barrier_data_, kBarrierDataSize, 0x00, kBarrierDataSize);
    if (ret != ACL_RT_SUCCESS) {
      MS_LOG(EXCEPTION) << "Failed to memset 0x00 for barrier data, ret = " << ret;
    }
  }
#endif
  return true;
}

bool HcomBarrierKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                               const std::vector<KernelTensor *> &, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    auto lccl_result =
      lccl_all_reduce_func_(lccl_ptr_, lccl_barrier_data_, lccl_barrier_data_, kBarrierDataSize / sizeof(int),
                            HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM, stream_ptr);
    if (lccl_result != Lcal_ms::LCAL_SUCCESS) {
      MS_LOG(EXCEPTION) << "LCCL AllReduce as Barrier failed.";
    }
    return true;
  }
#endif
  if (NeedReGetHcom()) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need re-get hcom";
    comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_);
  }
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBarrier(stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclBarrier failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

const std::vector<size_t> &HcomBarrierKernel::GetOutputSizeList() const {
  // Operators must have output, so give Barrier a dummy output.
  static const std::vector<size_t> dummy_output_size_list{kDim1 * sizeof(float)};
  return dummy_output_size_list;
}
}  // namespace kernel
}  // namespace mindspore
