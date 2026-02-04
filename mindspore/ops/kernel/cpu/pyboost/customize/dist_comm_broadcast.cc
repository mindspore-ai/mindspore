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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dist_comm_broadcast.h"
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
void DistCommBroadcastCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor, const Int64ImmPtr &src,
                                   const Int64ImmPtr &rank_id, const StringImmPtr &group) {
#if defined(__linux__) && defined(WITH_BACKEND)
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, tensor);
  op->set_outputs({tensor});
  auto src_rank = GetValue<int64_t>(src);
  auto local_rank = GetValue<int64_t>(rank_id);
  auto run_func = [op, tensor, src_rank, local_rank, group]() {
    auto device_context = op->device_context();
    if (local_rank == src_rank) {
      PyBoostUtils::MallocOpInputs(device_context, tensor);
    } else {
      PyBoostUtils::MallocOpOutputs(device_context, {tensor});
    }
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), tensor);
    auto in_addr = input_address_info.first;
    const auto &group_str = GetValue<std::string>(group);
    size_t type_len = GetDataTypeSize(in_addr[0]->dtype_id());
    auto comm_lib = distributed::collective::CollectiveManager::instance()->host_comm_lib();
    bool ret = comm_lib->Broadcast(in_addr[0]->device_ptr(), in_addr[0]->device_ptr(), in_addr[0]->size() / type_len,
                                   in_addr[0]->dtype_id(), src_rank, group_str);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Broadcast failed.";
    }
  };
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(run_func));
#else
  MS_LOG(EXCEPTION) << "The CPU op broadcast is only supported on linux platform.";
#endif
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
