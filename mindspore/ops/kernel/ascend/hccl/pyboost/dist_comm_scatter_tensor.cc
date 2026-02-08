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

#include "kernel/ascend/hccl/pyboost/dist_comm_scatter_tensor.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommScatterTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                          const TensorPtr &input_tensor, const Int64ImmPtr &rank_size,
                                          const Int64ImmPtr &src, const Int64ImmPtr &rank_id,
                                          const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, rank_size, src, rank_id, group);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, rank_size, src, group]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, op->outputs());

    auto rank_size_imm = GetValue<int64_t>(rank_size);
    auto [hccl_count, hccl_data_type] =
      HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor, rank_size_imm);
    auto root = GetValue<int64_t>(src);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, root](const HcclComm &hccl_comm,
                                                                                           void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclScatter(input_data_ptr, output_data_ptr, hccl_count,
                                                                      hccl_data_type, root, hccl_comm, comm_stream_ptr);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclScatter failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr, root);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
