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

#include "kernel/ascend/hccl/pyboost/dist_comm_all_to_all_v_c.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommAllToAllVCAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                       const TensorPtr &input_tensor, const StringImmPtr &group,
                                       const ValueTuplePtr &send_count_matrix, const Int64ImmPtr &rank_size,
                                       const Int64ImmPtr &rank_id) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, group, send_count_matrix, rank_size, rank_id);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, group, send_count_matrix, rank_size, rank_id]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, {other_tensor});
    auto hccl_data_type = HcomUtil::ConvertHcclType(input_tensor->data_type());
    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto rank_size_imm = GetValue<int64_t>(rank_size);
    const auto &send_count_matrix_vec = ConvertValueTupleToVector<int64_t>(send_count_matrix);
    std::shared_ptr<int64_t[]> params = std::shared_ptr<int64_t[]>(new int64_t[rank_size_imm * rank_size_imm]);
    for (int64_t i = 0; i < rank_size_imm; ++i) {
      for (int64_t j = 0; j < rank_size_imm; ++j) {
        params[i * rank_size_imm + j] = send_count_matrix_vec[i * rank_size_imm + j];
      }
    }
    auto launch_func = [input_data_ptr, output_data_ptr, params, hccl_data_type](const HcclComm &hccl_comm,
                                                                                 void *comm_stream_ptr) {
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAlltoAllVC(
        input_data_ptr, reinterpret_cast<void *>(params.get()), hccl_data_type, output_data_ptr, hccl_data_type,
        comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclAlltoAllVC failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
