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

#include "kernel/ascend/hccl/pyboost/dist_comm_broadcast.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommBroadcastAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor,
                                      const Int64ImmPtr &src, const Int64ImmPtr &rank_id, const StringImmPtr &group) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, tensor);
  op->set_outputs({tensor});

  auto run_func = [op, tensor, src, rank_id, group]() {
    auto device_context = op->device_context();
    auto src_imm = GetValue<int64_t>(src);
    auto local_rank = GetValue<int64_t>(rank_id);
    if (local_rank == src_imm) {
      PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, tensor);
    } else {
      PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, {tensor});
    }

    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), tensor);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, tensor);
    auto launch_func = [input_data_ptr, hccl_count, hccl_data_type, src_imm](const HcclComm &hccl_comm,
                                                                             void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBroadcast(input_data_ptr, hccl_count, hccl_data_type,
                                                                        src_imm, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclBroadcast failed, ret:" << hccl_result;
      }
    };
    auto post_func = [tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, tensor);
    };
    CommonCommAscendFunc(op, tensor, group, launch_func, post_func, src_imm);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
