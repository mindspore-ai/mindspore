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

#include "kernel/ascend/hccl/pyboost/inner_comm_irecv.h"

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
void InnerCommIrecvAscendCustomize(const std::shared_ptr<OpRunner> &op, const Int64ImmPtr &tag, const Int64ImmPtr &src,
                                   const ValueTuplePtr &shape, const StringImmPtr &group,
                                   const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, tag, src, shape, group, dtype);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, src, group]() {
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, op->outputs());

    const auto &output_tensor = op->output(0);
    auto hccl_count = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), output_tensor).first;
    auto src_imm = GetValue<int64_t>(src);

    auto output_data_ptr = GetDevicePtrFromTensor(op->primitive()->name(), output_tensor);
    auto output_dtype = output_tensor->data_type();
    std::string group_name = GetValue<std::string>(group);
    auto launch_func = [output_data_ptr, hccl_count, output_dtype, src_imm, group_name](const HcclComm &,
                                                                                        void *comm_stream_ptr) {
      auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
      auto comm_result =
        comm_lib->Recv(output_data_ptr, hccl_count, output_dtype, src_imm, group_name, comm_stream_ptr);
      if (!comm_result) {
        MS_LOG(EXCEPTION) << "InnerCommIrecv failed, ret:" << comm_result;
      }
    };

    auto post_func = [output_tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, output_tensor);
    };
    CommonCommAscendFunc(op, nullptr, group, launch_func, post_func, src_imm);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
