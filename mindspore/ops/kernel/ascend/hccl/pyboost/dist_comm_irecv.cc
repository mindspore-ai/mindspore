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

#include "kernel/ascend/hccl/pyboost/dist_comm_irecv.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommIrecvAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                  const Int64ImmPtr &tag, const Int64ImmPtr &src, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, tag, src, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  op->set_outputs({input_tensor});

  auto run_func = [op, input_tensor, src, group]() {
    // Malloc for input tensors
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, op->outputs());

    const auto &output_tensor = op->output(0);
    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), output_tensor);
    auto src_imm = GetValue<int64_t>(src);

    auto output_data_ptr = GetDevicePtrFromTensor(op->primitive()->name(), output_tensor);
    auto launch_func = [output_data_ptr, hccl_count, hccl_data_type, src_imm](const HcclComm &hccl_comm,
                                                                              void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclRecv(output_data_ptr, hccl_count, hccl_data_type, src_imm,
                                                                   comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcomRecv failed, ret:" << hccl_result;
      }
    };

    auto post_func = [output_tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, output_tensor);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func, src_imm);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
