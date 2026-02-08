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

#include "kernel/ascend/hccl/pyboost/inner_comm_isend.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommIsendAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                   const Int64ImmPtr &dst, const StringImmPtr &group, const Int64ImmPtr &tag) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);

  op->set_outputs({input_tensor});
  op->CreateOutputSimpleInfo();

  auto run_func = [op, input_tensor, dst, group]() {
    auto device_context = op->device_context();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);

    auto dst_imm = GetValue<int64_t>(dst);

    auto hccl_count = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor).first;
    auto input_data_ptr = GetDevicePtrFromTensor(op->primitive()->name(), input_tensor);
    auto input_dtype = input_tensor->data_type();
    std::string group_name = GetValue<std::string>(group);
    auto launch_func = [input_data_ptr, hccl_count, input_dtype, dst_imm, group_name](const HcclComm &,
                                                                                      void *comm_stream_ptr) {
      auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
      auto comm_result = comm_lib->Send(input_data_ptr, hccl_count, input_dtype, dst_imm, group_name, comm_stream_ptr);
      if (!comm_result) {
        MS_LOG(EXCEPTION) << "InnerCommIsend failed, ret:" << comm_result;
      }
    };

    auto post_func = [device_context, input_tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), device_context,
                                                                      comm_stream_id, event, input_tensor);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func, dst_imm);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
