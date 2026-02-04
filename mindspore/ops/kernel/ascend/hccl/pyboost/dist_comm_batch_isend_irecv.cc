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

#include "kernel/ascend/hccl/pyboost/dist_comm_batch_isend_irecv.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommBatchIsendIrecvAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &input_tensor,
                                            const StringImmPtr &group, const ValueTuplePtr &op_types,
                                            const ValueTuplePtr &remotes_ranks) {
  const auto &op_types_list = ConvertValueTupleToVector<int64_t>(op_types);
  const auto &remotes_rank_list = ConvertValueTupleToVector<int64_t>(remotes_ranks);
  std::vector<TensorPtr> input_tensors = ConvertValueTupleToVector<TensorPtr>(input_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensors);
  op->set_outputs({input_tensors[op_types_list.size() - 1]});

  auto run_func = [op, input_tensors, group, op_types_list, remotes_rank_list]() {
    auto item_num = static_cast<uint32_t>(op_types_list.size());
    PyBoostUtils::MallocOpInputsWithStream(op->device_context(), kDefaultStreamIndex, input_tensors);

    auto launch_func = [op, input_tensors, remotes_rank_list, op_types_list, item_num](const HcclComm &hccl_comm,
                                                                                       void *comm_stream_ptr) {
      HcclSendRecvItem *send_recv_info = new HcclSendRecvItem[item_num];
      for (size_t i = 0; i < item_num; ++i) {
        HcclSendRecvType type;
        if (op_types_list[i] == 0) {
          type = HcclSendRecvType::HCCL_SEND;
        } else if (op_types_list[i] == 1) {
          type = HcclSendRecvType::HCCL_RECV;
        } else {
          type = HcclSendRecvType::HCCL_SEND_RECV_RESERVED;
        }
        auto buf = GetDevicePtrFromTensor(op->primitive()->name(), input_tensors[i]);
        auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensors[i]);
        auto rank = static_cast<uint32_t>(remotes_rank_list[i]);
        send_recv_info[i] = HcclSendRecvItem{type, buf, hccl_count, hccl_data_type, rank};
      }

      auto hccl_result =
        hccl::HcclAdapter::GetInstance().HcclBatchISendIRecv(send_recv_info, item_num, hccl_comm, comm_stream_ptr);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclBatchISendIRecv failed, ret:" << hccl_result;
      }
      delete[] send_recv_info;
    };

    auto post_func = [input_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensors);
    };
    CommonCommAscendFunc(op, input_tensors[0], group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
