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

#include "kernel/ascend/hccl/pyboost/comm_common.h"
#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/framework_ops.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/backend/common/exec_order/execute_order_tracker.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/pynative/utils/pyboost/comm_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/backend/common/exec_order/kernel_cache.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void SimuExecSetOutput(const tensor::TensorPtr &output_tensor, const std::string &op_name) {
  auto output_size = output_tensor->Size();
  auto output_type = static_cast<TypeId>(output_tensor->data_type_c());
  static const float kInitValue = 0.1f;
  static const size_t kFp32TypeSize = abstract::TypeIdSize(kNumberTypeFloat32);
  std::vector<float> init_value;
  std::vector<float> host_data;
  init_value.resize(output_size, kInitValue);
  host_data.resize(output_size, 0);
  void *host_ptr = init_value.data();
  if (output_type != kNumberTypeFloat32) {
    auto elem_num = output_size / abstract::TypeIdSize(output_type);
    const trans::TypeIdArgs type_args{init_value.data(), SizeToLong(elem_num), kNumberTypeFloat32, output_type,
                                      elem_num * kFp32TypeSize};
    auto sync_ok = trans::TransDataType(type_args, host_data.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "simu SimuExecSetOutput trans data type failed.";
      return;
    }
    host_ptr = host_data.data();
  }

  auto data_ptr = GetDevicePtrFromTensor(op_name, output_tensor);
  auto cp_ret = CALL_ASCEND_API(aclrtMemcpy, data_ptr, output_size, host_ptr, output_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "Simu receive memset 0 failed.";
    return;
  }
}

void CommonCommRunTask(const std::function<void(void)> &run_func) {
  if (runtime::OpExecutor::NeedSync()) {
    run_func();
  } else {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
      std::make_shared<runtime::PassthroughNoWaitDeviceTask>(run_func));
  }
}

void CommonCommAscendFunc(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const StringImmPtr &group,
                          const std::function<void(const HcclComm &, void *)> &launch_func,
                          const std::function<void(const DeviceEventPtr &, size_t)> &post_func, int64_t rank) {
  const auto &op_name = op->primitive()->name();
  MS_LOG(DEBUG) << "Run device task " << op_name << " end";

  const auto &group_str = GetValue<std::string>(group);
  // Before calling each hccl operator, we need to wait for communicator to be initialized.
  distributed::collective::CollectiveManager::instance()->WaitCommInitDone(group_str);
  const auto &hccl_comm = device::ascend::AscendCollectiveCommLib::GetInstance().GetHcomByGroup(group_str);
  auto comm_handle = op->comm_handle();
  auto device_context = op->device_context();
  static auto sync = runtime::RuntimeConf::GetInstance()->launch_blocking();

  // Need to bind context if the comm_op is the first op launched in this thread.
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  size_t comm_stream_id = 0;
  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMultiStream)) {
    // multi_stream:true, all communication op use the same communication stream
    comm_stream_id = device_context->device_res_manager_->GetCommunicationStreamID();
  } else if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeMultiStream)) {
    // multi_stream:false, all communication op use the same op stream
    comm_stream_id = op->stream_id();
  } else {
    // Default scene, multi_stream:group, all communication op use the communication stream by group
    comm_stream_id = device_context->device_res_manager_->GetCommunicationStreamIDByGroup(group_str);
  }

  auto func = [device_context, group_str, op_stream_id = op->stream_id(), comm_handle, hccl_comm, comm_stream_id,
               op_name, launch_func]() {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask,
                                       op_name, false);

    device::tracker::CALL_MEMORY_TRACKER(UpdateTask, "PyNative",
                                         {{device::tracker::kStreamId, std::to_string(comm_stream_id)}});
    device::tracker::CALL_MEMORY_TRACKER(CacheLastTask);
    CommUtils::GetInstance().SyncOpStream(device_context, op_stream_id, comm_stream_id);
    device::tracker::CALL_MEMORY_TRACKER(EmptyCache);

    auto comm_stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(comm_stream_id);

    if (launch_func) {
      launch_func(hccl_comm, comm_stream_ptr);
      if (sync) {
        if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
          MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name;
        }
      }
    }
    comm_handle->RecordEvent(comm_stream_id);
  };

  static bool dry_run = common::IsExecuteSimulation();
  if (MS_UNLIKELY(dry_run)) {
    SimuExecSetOutput(op->output(0), op->primitive()->name());
  }

  runtime::OpExecutor::DispatchLaunchTask(func);

  static runtime::KernelCache &cache = runtime::KernelCache::GetInstance();
  if (MS_UNLIKELY(cache.need_add)) {
    cache.AddPyboostKernel(op->primitive()->name(), group_str, tensor::ShapeToString(input_tensor->shape()),
                           tensor::ShapeToString(op->output(0)->shape()), rank);
  }
  if (MS_UNLIKELY(EnableExecuteOrderDump())) {
    auto &execute_order_tracker = ExecuteOrderTracker::GetInstance();
    execute_order_tracker.ProcessPyboostCommOp(op, group_str, comm_stream_id, input_tensor, op->output(0), rank);
  }

  if (post_func) {
    post_func(comm_handle->event(), comm_stream_id);
  } else if (input_tensor != nullptr) {
    // Default post function
    runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(
      op->primitive()->name(), device_context, comm_stream_id, comm_handle->event(), input_tensor, op->output(0));
  }
  comm_handle->UpdateTaskId(comm_stream_id);

  if (sync) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name;
    }
  }
  MS_LOG(DEBUG) << "Run device task " << op_name << " end";
}

void *GetDevicePtrFromTensor(const std::string &op_name, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);

  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  auto storage_info = tensor->storage_info();
  if (storage_info == nullptr) {
    return device_address->GetMutablePtr();
  }

  if (!storage_info->is_contiguous) {
    MS_EXCEPTION(ValueError) << op_name
                             << " does not support not-contiguous tensor. Please call tensor.contiguous() firstly.";
  }

  if (storage_info->storage_offset == 0) {
    return device_address->GetMutablePtr();
  }

  size_t offset = mindspore::abstract::TypeIdSize(tensor->data_type()) * storage_info->storage_offset;
  // tensor is contiguous, add offset for addr
  return reinterpret_cast<char *>(device_address->GetMutablePtr()) + offset;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
