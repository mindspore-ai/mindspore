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

#include "kernel/ascend/hccl/pyboost/inner_comm_reduce_scatter.h"

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
void InnerCommReduceScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                           const Int64ImmPtr &rank_size, const StringImmPtr &op_type,
                                           const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, rank_size, op_type, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, input_tensor, rank_size, op_type, group]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, op->outputs());

    auto rank_size_imm = GetValue<int64_t>(rank_size);
    auto hccl_count = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor, rank_size_imm).first;
    auto reduce_op = HcomUtil::GetCollectiveOpReduceType(GetValue<std::string>(op_type));

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto input_dtype = input_tensor->data_type();
    std::string group_name = GetValue<std::string>(group);
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, input_dtype, reduce_op, group_name](
                         const HcclComm &, void *comm_stream_ptr) {
      auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
      auto comm_result = comm_lib->ReduceScatter(input_data_ptr, output_data_ptr, hccl_count, input_dtype, reduce_op,
                                                 group_name, comm_stream_ptr);
      if (!comm_result) {
        MS_LOG(EXCEPTION) << "InnerCommReduceScatter failed, ret:" << comm_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
