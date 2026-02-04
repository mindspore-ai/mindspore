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

#include "kernel/ascend/hccl/pyboost/dist_comm_barrier.h"

#include <memory>
#include <string>
#include "ir/tensor_new.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommBarrierAscendCustomize(const std::shared_ptr<OpRunner> &op, const StringImmPtr &group) {
  auto op_type_enum = HCCL_REDUCE_SUM;
  TensorPtr input_tensor = tensor::from_scalar(0.0);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  op->set_outputs({input_tensor});
  auto run_func = [op, input_tensor, op_type_enum, group]() {
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, {input_tensor});
    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto launch_func = [input_data_ptr, hccl_count, hccl_data_type, op_type_enum](const HcclComm &hccl_comm,
                                                                                  void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllReduce(
        input_data_ptr, input_data_ptr, hccl_count, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclAllReduce failed, ret:" << hccl_result;
      }
    };
    auto post_func = [input_tensor, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
