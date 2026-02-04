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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dist_comm_gather.h"
#include <memory>
#include <utility>
#include <string>
#include "include/pynative/utils/pyboost/customize/op_common.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "utils/misc.h"
#endif

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommGatherCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                const ValueTuplePtr &gather_list, const Int64ImmPtr &rank_size, const Int64ImmPtr &dst,
                                const Int64ImmPtr &rank_id, const StringImmPtr &group) {
#if defined(__linux__) && defined(WITH_BACKEND)
  OpRunner::InferOpOutput(op, input_tensor, gather_list, rank_size, dst, rank_id, group);
  auto dst_rank = GetValue<int64_t>(dst);
  auto local_rank = GetValue<int64_t>(rank_id);
  auto rank_size_imm = static_cast<size_t>(GetValue<int64_t>(rank_size));
  std::vector<TensorPtr> gather_tensors = ConvertValueTupleToVector<TensorPtr>(gather_list);
  if (local_rank == dst_rank) {
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, gather_tensors);
  } else {
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  }
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto run_func = [op, gather_tensors, input_tensor, local_rank, dst_rank, group, rank_size_imm]() {
    PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
    PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
    if (local_rank == dst_rank) {
      PyBoostUtils::MallocOpOutputs(op->device_context(), gather_tensors);
    }

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(op->device_context(), op->stream_id(), op->input_abs(), input_tensor);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(op->device_context(), op->stream_id(), {op->output_abs()}, op->outputs());

    auto in_addr = input_address_info.first;
    auto data_size = in_addr[0]->size();
    auto out_addr = output_address_info.first;
    const auto &group_str = GetValue<std::string>(group);
    auto type_len = GetDataTypeSize(in_addr[0]->dtype_id());

    auto comm_lib = distributed::collective::CollectiveManager::instance()->host_comm_lib();
    bool ret = comm_lib->Gather(in_addr[0]->device_ptr(), out_addr[0]->device_ptr(), data_size / type_len,
                                in_addr[0]->dtype_id(), dst_rank, group_str);
    if (!ret) {
      MS_LOG(EXCEPTION) << "AllGather failed.";
    }
    if (local_rank == dst_rank) {
      const auto &gather_address_info =
        PyBoostUtils::GetAddressInfo(op->device_context(), op->stream_id(), op->input_abs(), gather_tensors);
      for (size_t r = 0; r < rank_size_imm; r++) {
        auto gather_addr = (gather_address_info.first)[r]->device_ptr();
        auto output_addr = out_addr[0]->device_ptr();
        auto offset = static_cast<size_t>(r * data_size);
        auto mem_ret = Memcpy(reinterpret_cast<char *>(gather_addr), data_size,
                              reinterpret_cast<char *>(output_addr) + offset, data_size);
        if (mem_ret != EOK) {
          MS_LOG(EXCEPTION) << "Memcpy failed. ret is " << mem_ret;
        }
      }
    }
  };

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(run_func));
#else
  MS_LOG(EXCEPTION) << "The CPU op gather is only supported on linux platform.";
#endif
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
