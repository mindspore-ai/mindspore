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

#include "kernel/ascend/hccl/pyboost/dist_comm_all_gather.h"

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
void DistCommAllGatherAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &gather_list,
                                      const TensorPtr &input_tensor, const Int64ImmPtr &rank_size,
                                      const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, gather_list, input_tensor, rank_size, group);

  std::vector<TensorPtr> gather_tensors = ConvertValueTupleToVector<TensorPtr>(gather_list);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, gather_tensors, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());
  auto rank_size_imm = GetValue<int64_t>(rank_size);
  auto run_func = [op, gather_tensors, input_tensor, group, rank_size_imm]() {
    PyBoostUtils::MallocOpInputsWithStream(op->device_context(), kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, gather_tensors);
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, op->outputs());
    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto size = input_tensor->Size();
    auto input_num_elements = input_tensor->DataSize();

    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, rank_size_imm, size,
                        input_num_elements, gather_tensors, op_name](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      bool same_shape = std::all_of(gather_tensors.begin(), gather_tensors.end(),
                                    [&](const TensorPtr &t) { return t->Size() == gather_tensors[0]->Size(); });
      if (same_shape) {
        auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(input_data_ptr, output_data_ptr, hccl_count,
                                                                          hccl_data_type, comm_stream_ptr, hccl_comm);
        if (hccl_result != HCCL_SUCCESS) {
          MS_LOG(EXCEPTION) << "HcclAllGather failed, ret:" << hccl_result;
        }
        for (int r = 0; r < rank_size_imm; r++) {
          uint64_t offset = static_cast<uint64_t>(r * size);
          auto data_ptr = GetDevicePtrFromTensor(op_name, gather_tensors[r]);
          auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, size, static_cast<char *>(output_data_ptr) + offset,
                                        size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
          }
        }
      } else {
        MS_LOG(DEBUG) << "For kernel HcclAllGather, Different shapes detected, using HcclAllGatherV instead.";
        hccl::HcclAllGatherVParams params;
        std::vector<uint64_t> recv_counts;
        std::vector<uint64_t> recv_displs;
        std::vector<uint64_t> recv_size_byte;
        std::vector<uint64_t> recv_offset_byte;
        params.send_count = static_cast<uint64_t>(input_num_elements);

        recv_counts.reserve(rank_size_imm);
        recv_displs.resize(rank_size_imm, 0);
        recv_offset_byte.resize(rank_size_imm, 0);
        for (const auto &tensor : gather_tensors) {
          recv_counts.push_back(static_cast<uint64_t>(tensor->DataSize()));
          recv_size_byte.push_back(static_cast<uint64_t>(tensor->Size()));
        }
        for (int i = 1; i < rank_size_imm; ++i) {
          recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
          recv_offset_byte[i] = recv_offset_byte[i - 1] + recv_size_byte[i - 1];
        }

        params.recv_counts = recv_counts;
        params.rdispls = recv_displs;

        auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGatherV(input_data_ptr, output_data_ptr, params,
                                                                           hccl_data_type, comm_stream_ptr, hccl_comm);
        if (hccl_result != HCCL_SUCCESS) {
          MS_LOG(EXCEPTION) << "HcclAllGatherV failed, ret:" << hccl_result;
        }

        for (int r = 0; r < rank_size_imm; r++) {
          uint64_t offset = recv_offset_byte[r];
          auto copy_size = recv_size_byte[r];
          if (copy_size == 0) {
            continue;
          }
          auto data_ptr = GetDevicePtrFromTensor(op_name, gather_tensors[r]);
          auto cp_ret =
            CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, copy_size, static_cast<char *>(output_data_ptr) + offset,
                            copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "HcclAllGather aclrtMemcpy failed.";
          }
        }
      }
    };
    auto post_func = [input_tensor, gather_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor,
                                                                      op->output(0), gather_tensors);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
