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

#include "kernel/ascend/hccl/pyboost/dist_comm_reduce_scatter_tensor.h"

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
void DistCommReduceScatterTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                                const TensorPtr &input_tensor, const Int64ImmPtr &rank_size,
                                                const StringImmPtr &op_type, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, rank_size, op_type, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, rank_size, op_type, group]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, op->outputs());
    auto rank_size_imm = GetValue<int64_t>(rank_size);
    auto [hccl_count, hccl_data_type] =
      HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor, rank_size_imm);
    auto op_type_str = GetValue<std::string>(op_type);
    bool is_avg = (op_type_str == "avg");
    auto real_op_type = is_avg ? "sum" : op_type_str;
    auto op_type_enum = HcomUtil::GetHcomReduceOpType(real_op_type);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto output_tensor = op->output(0);
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum](
                         const HcclComm &hccl_comm, void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatter(
        input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclReduceScatter failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr, /*rank=*/-1,
                         /*avg_div_tensor=*/is_avg ? output_tensor : nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
