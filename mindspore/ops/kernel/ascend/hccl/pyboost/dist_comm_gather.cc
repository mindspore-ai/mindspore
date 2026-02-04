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

#include "kernel/ascend/hccl/pyboost/dist_comm_gather.h"

#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommGatherAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                   const ValueTuplePtr &gather_list, const Int64ImmPtr &rank_size,
                                   const Int64ImmPtr &dst, const Int64ImmPtr &rank_id, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, gather_list, rank_size, dst, rank_id, group);
  auto dst_rank = GetValue<int64_t>(dst);
  auto local_rank = GetValue<int64_t>(rank_id);
  auto rank_size_imm = GetValue<int64_t>(rank_size);
  std::vector<TensorPtr> gather_tensors = ConvertValueTupleToVector<TensorPtr>(gather_list);
  if (local_rank == dst_rank) {
    PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor, gather_tensors);
  } else {
    PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  }
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, gather_tensors, input_tensor, local_rank, dst_rank, group, rank_size_imm, dst]() {
    PyBoostUtils::MallocOpInputsWithStream(op->device_context(), kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, op->outputs());
    if (local_rank == dst_rank) {
      PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, gather_tensors);
    }

    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);
    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));

    auto size = input_tensor->Size();
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, local_rank, dst_rank,
                        rank_size_imm, size, gather_tensors,
                        op_name](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(input_data_ptr, output_data_ptr, hccl_count,
                                                                        hccl_data_type, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcomGather failed, ret:" << hccl_result;
      }
      if (local_rank == dst_rank) {
        for (int r = 0; r < rank_size_imm; r++) {
          uint64_t offset = (uint64_t)(r * size);
          auto data_ptr = GetDevicePtrFromTensor(op_name, gather_tensors[r]);

          auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, size, static_cast<char *>(output_data_ptr) + offset,
                                        size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
          }
        }
      }
    };
    auto post_func = [input_tensor, gather_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor,
                                                                      op->output(0), gather_tensors);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func, dst_rank);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
