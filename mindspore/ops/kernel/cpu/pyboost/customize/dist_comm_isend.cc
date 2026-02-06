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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dist_comm_isend.h"
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
void DistCommIsendCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                               const Int64ImmPtr &dst, const StringImmPtr &group, const Int64ImmPtr &tag) {
#if defined(__linux__) && defined(WITH_BACKEND)
  MS_LOG(DEBUG) << "Call start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});

  auto run_func = [op, input_tensor, dst, group]() {
    auto dst_rank = GetValue<int64_t>(dst);
    PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(op->device_context(), op->stream_id(), op->input_abs(), input_tensor);

    auto in_addr = input_address_info.first;
    const auto &group_str = GetValue<std::string>(group);
    size_t type_len = GetDataTypeSize(in_addr[0]->dtype_id());
    auto comm_lib = distributed::collective::CollectiveManager::instance()->host_comm_lib();
    bool ret = comm_lib->Send(in_addr[0]->device_ptr(), in_addr[0]->size() / type_len, in_addr[0]->dtype_id(), dst_rank,
                              group_str);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Send failed.";
    }
  };

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(run_func));
#else
  MS_LOG(EXCEPTION) << "The CPU op scatter is only supported on linux platform.";
#endif
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
