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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dist_comm_all_reduce.h"
#include <memory>
#include <utility>
#include <string>
#include <unordered_map>
#include "include/pynative/utils/pyboost/customize/op_common.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "utils/misc.h"
#endif

namespace mindspore {
namespace kernel {
#if defined(__linux__) && defined(WITH_BACKEND)
using device::CollectiveOpReduceType;
#endif
namespace pyboost {
void DistCommAllReduceCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor,
                                   const StringImmPtr &op_type, const StringImmPtr &group) {
#if defined(__linux__) && defined(WITH_BACKEND)
  MS_LOG(DEBUG) << "Call start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, tensor);
  op->set_outputs({tensor});

  std::unordered_map<std::string, CollectiveOpReduceType> kHcomReduceOpTypeMap = {
    {"min", CollectiveOpReduceType::Reduce_Min},
    {"max", CollectiveOpReduceType::Reduce_Max},
    {"prod", CollectiveOpReduceType::Reduce_Prod},
    {"sum", CollectiveOpReduceType::Reduce_Sum},
  };
  auto iter = kHcomReduceOpTypeMap.find(GetValue<std::string>(op_type));
  if (iter == kHcomReduceOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "Get HCOM_ATTR_REDUCE_TYPE fail, [" << GetValue<std::string>(op_type) << "] not support!";
  }
  auto op_type_enum = iter->second;

  auto run_func = [op, tensor, op_type_enum, group]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputs(device_context, tensor);
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), tensor);
    auto in_addr = input_address_info.first;
    const auto &group_str = GetValue<std::string>(group);
    size_t type_len = GetDataTypeSize(in_addr[0]->dtype_id());
    auto comm_lib = distributed::collective::CollectiveManager::instance()->host_comm_lib();
    bool ret = comm_lib->AllReduce(in_addr[0]->device_ptr(), in_addr[0]->device_ptr(), in_addr[0]->size() / type_len,
                                   in_addr[0]->dtype_id(), op_type_enum, group_str);
    if (!ret) {
      MS_LOG(EXCEPTION) << "AllReduce failed.";
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
