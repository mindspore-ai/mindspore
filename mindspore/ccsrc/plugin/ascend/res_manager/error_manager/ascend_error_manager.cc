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

#include "plugin/ascend/res_manager/error_manager/ascend_error_manager.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "device_address/device_type.h"
#include "ir/tensor_new.h"
#include "plugin/ascend/res_manager/error_manager/param_restore.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/ascend_res_manager.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;

int TftGetErrorCode() {
  auto aclrt_get_last_error = mindspore::device::ascend::aclrtGetLastError_;
  MS_EXCEPTION_IF_NULL(aclrt_get_last_error);
  return aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
}

const char *TftGetRecentErrMsg() {
  auto acl_get_recent_err_msg = mindspore::device::ascend::aclGetRecentErrMsg_;
  MS_EXCEPTION_IF_NULL(acl_get_recent_err_msg);
  return acl_get_recent_err_msg();
}

REGISTER_COMMON_CALLBACK(TftGetErrorCode);
REGISTER_COMMON_CALLBACK(TftGetRecentErrMsg);

bool IsDeviceMemError(int error_code) { return error_code == ACL_ERROR_RT_DEVICE_MEM_ERROR; }
bool IsHbmMultBitEccError(int error_code) { return error_code == ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR; }
bool IsCommOpRetryFailError(int error_code) { return error_code == ACL_ERROR_RT_COMM_OP_RETRY_FAIL; }
bool IsForceStopError(int error_code) { return error_code == ACL_ERROR_RT_DEVICE_TASK_ABORT; }
bool IsSuspectRemoteError(int error_code) { return error_code == ACL_ERROR_RT_SUSPECT_REMOTE_ERROR; }

REGISTER_COMMON_CALLBACK(IsDeviceMemError);
REGISTER_COMMON_CALLBACK(IsHbmMultBitEccError);
REGISTER_COMMON_CALLBACK(IsCommOpRetryFailError);
REGISTER_COMMON_CALLBACK(IsForceStopError);
REGISTER_COMMON_CALLBACK(IsSuspectRemoteError);

int TftSendRecvParams(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank, bool use_batch) {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  auto ascend_res_manager = static_cast<device::ascend::AscendResManager *>(device_ctx->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(ascend_res_manager);
  ParamReplication replicator(ascend_res_manager);
  replicator.Init();
  return replicator.SendRecv(params, src_rank, dst_rank, use_batch);
}

REGISTER_COMMON_CALLBACK(TftSendRecvParams);

std::pair<uint64_t, uint64_t> TftGetOptimizerTimestamps() {
  OptimizerEventInfo::GetInstance().GetOptimizerTimestamp(false);
  auto opt_start_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_start_timestamp();
  auto opt_end_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_end_timestamp();
  return std::pair<uint64_t, uint64_t>{opt_start_timestamp, opt_end_timestamp};
}

REGISTER_COMMON_CALLBACK(TftGetOptimizerTimestamps);

void TftSaveParameters(const std::vector<AnfNodePtr> &weights, aclrtStream stream,
                       std::map<std::string, tensor::TensorPtr> *ptr_params) {
  MS_EXCEPTION_IF_NULL(ptr_params);
  auto &saved_params = *ptr_params;
  int index = 0;
  for (const auto &node : weights) {
    index += 1;
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto out_kernel_tensor = session::AnfRuntimeAlgorithm::GetOutputKernelTensor(param, 0, false);
      if (out_kernel_tensor == nullptr || out_kernel_tensor->device_address() == nullptr ||
          out_kernel_tensor->device_address()->GetPtr() == nullptr ||
          IsOneOfHWSpecialFormat(kernel::GetFormatFromEnumToStr(out_kernel_tensor->format()))) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto iter = saved_params.find(param->name());
      if (iter == saved_params.end()) {
        MS_LOG(WARNING) << "Can not find parameter " << param->name() << " in saved parameters.";
        continue;
      }
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (iter->second == nullptr) {
        // NOTE: here use `pin_mem_allocator` to allocate host memory for saving snapshot, otherwise there would be more
        // overhead when copying data from device to host
        const auto &shape = tensor->shape();
        const auto &dtype = tensor->data_type();
        auto device_address = DeviceAddressMaker(nullptr, dtype, shape)
                                .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                                .make_device_address();

        auto ascend_device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(
          device::GetDeviceNameByType(device::DeviceType::kAscend));
        if (ascend_device_ctx == nullptr || ascend_device_ctx->device_res_manager_ == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot find Ascend device context. ascend_device_ctx or device_res_manager is null.";
        }

        auto pin_memory_allocator = ascend_device_ctx->device_res_manager_->pin_mem_allocator();
        std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->set_allocator(pin_memory_allocator);

        auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(
          device::GetDeviceNameByType(device::DeviceType::kCPU));
        bool allocate_mem_ret = device_ctx->device_res_manager_->AllocateMemory(
          std::dynamic_pointer_cast<device::DeviceAddress>(device_address).get());
        if (!allocate_mem_ret) {
          MS_LOG(EXCEPTION) << "Tensor.pin_memory allocate memory failed!";
        }

        auto tensor_ptr = std::make_shared<tensor::Tensor>(dtype, shape, device_address);
        saved_params[param->name()] = tensor_ptr;
      }
      auto host_tensor = saved_params[param->name()];
      auto size = tensor->Size();
      MS_LOG(INFO) << "Copy parameter " << param->name() << " with size " << size << " " << index << "/"
                   << weights.size();
      auto ret =
        CALL_ASCEND_API(aclrtMemcpyAsync, host_tensor->data_c(), size,
                        out_kernel_tensor->device_address()->GetMutablePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
    }
  }

  // record event to notify whether the optimizer can be run
  SnapshotHelper::GetInstance().RecordEvent(stream);
}

REGISTER_COMMON_CALLBACK(TftSaveParameters);

bool TftProcessOptimizerEvent(const CNodePtr &kernel, KernelMod *kernel_mod, void *stream, bool is_saving_snapshot,
                              bool is_enable_uce) {
  // async ckpt and weight snapshot
  auto opt_start_type = OptimizerEventInfo::GetInstance().GetOptimizerStartType(kernel_mod, kernel);
  bool is_opt_start_kernel = (opt_start_type != OptStartType::OPT_START_TYPE_NONE);
  if (MS_UNLIKELY(is_opt_start_kernel && is_saving_snapshot)) {
    SnapshotHelper::GetInstance().StreamWaitEvent(stream);
  }
  if (opt_start_type == OptStartType::OPT_START_TYPE_SNAPSHOT) {
    // skip execute TensorReport op with attribute "snapshot", it is just used as a tag
    return true;
  }

  bool is_opt_end_kernel = OptimizerEventInfo::GetInstance().IsOptimizerEndKernelMod(kernel_mod, kernel);
  if (is_enable_uce) {
    if (is_opt_start_kernel || is_opt_end_kernel) {
      // insert event for optimizer start and end
      OptimizerEventInfo::GetInstance().RecordEvent(is_opt_start_kernel, stream);
    }
  }
  if (is_opt_end_kernel) {
    // skip execute TensorReport op at the end of optimizer, it is just used as a tag
    return true;
  }
  return false;
}

REGISTER_COMMON_CALLBACK(TftProcessOptimizerEvent);

void ResetOptimizerEventInfo() { OptimizerEventInfo::GetInstance().Reset(); }

REGISTER_COMMON_CALLBACK(ResetOptimizerEventInfo);

void DestroySnapshotHelper() { SnapshotHelper::GetInstance().Clear(); }

REGISTER_COMMON_CALLBACK(DestroySnapshotHelper);

SnapshotHelper &SnapshotHelper::GetInstance() {
  static SnapshotHelper instance;
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (instance.async_copy_event_ == nullptr) {
      if (CALL_ASCEND_API(aclrtCreateEventExWithFlag, &instance.async_copy_event_, ACL_EVENT_SYNC) != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Create async event failed";
      }
    }
  });
  return instance;
}

void SnapshotHelper::Clear() {
  if (async_copy_event_ != nullptr) {
    auto ret = CALL_ASCEND_API(aclrtDestroyEvent, async_copy_event_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtDestroyEvent failed with return value " << ret;
    }
    async_copy_event_ = nullptr;
  }
}

SnapshotHelper::~SnapshotHelper() { Clear(); }

void SnapshotHelper::RecordEvent(aclrtStream stream) {
  aclError ret = CALL_ASCEND_API(aclrtRecordEvent, async_copy_event_, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtRecordEvent failed, error code is " << ret;
  }
}

void SnapshotHelper::StreamWaitEvent(aclrtStream stream) {
  aclError ret = CALL_ASCEND_API(aclrtStreamWaitEvent, stream, async_copy_event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtStreamWaitEvent failed, error code is " << ret;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
