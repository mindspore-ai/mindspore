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

#include "kernel/ascend/hccl/pyboost/dist_comm_all_to_all_v_single.h"

#include <memory>
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {

void GetAllToAllVParam(const ValueTuplePtr &send_numel_list, const ValueTuplePtr &recv_numel_list,
                       hccl::HcclAllToAllVParams *params) {
  const auto &send_numel_list_vec = ConvertValueTupleToVector<int64_t>(send_numel_list);
  const auto &recv_numel_list_vec = ConvertValueTupleToVector<int64_t>(recv_numel_list);

  uint64_t offset = 0;
  for (size_t i = 0; i < send_numel_list_vec.size(); i++) {
    auto count = static_cast<uint64_t>(send_numel_list_vec[i]);
    params->sendcounts.push_back(count);
    params->sdispls.push_back(offset);
    offset += count;
  }
  offset = 0;
  for (size_t i = 0; i < recv_numel_list_vec.size(); i++) {
    auto count = static_cast<uint64_t>(recv_numel_list_vec[i]);
    params->recvcounts.push_back(count);
    params->rdispls.push_back(offset);
    offset += count;
  }
}

std::function<void(const HcclComm &, void *)> CallAllToAllV(const std::shared_ptr<OpRunner> &op,
                                                            const TensorPtr &input_tensor,
                                                            const ValueTuplePtr &send_numel_list,
                                                            const ValueTuplePtr &recv_numel_list) {
  MS_LOG(DEBUG) << "Begin";
  hccl::HcclAllToAllVParams params;
  GetAllToAllVParam(send_numel_list, recv_numel_list, &params);
  auto hccl_data_type = HcomUtil::ConvertHcclType(input_tensor->data_type());
  const auto &op_name = op->primitive()->name();
  auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
  auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
  auto launch_func = [input_data_ptr, output_data_ptr, hccl_data_type, params](const HcclComm &hccl_comm,
                                                                               void *comm_stream_ptr) {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAlltoAllV(input_data_ptr, output_data_ptr, params,
                                                                      hccl_data_type, comm_stream_ptr, hccl_comm);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(EXCEPTION) << "HcclAllToAllv failed, ret:" << hccl_result;
    }
  };
  return launch_func;
}

std::function<void(const HcclComm &, void *)> CallAllToAll(const std::shared_ptr<OpRunner> &op,
                                                           const TensorPtr &input_tensor,
                                                           const Int64ImmPtr &rank_size) {
  MS_LOG(DEBUG) << "Begin";
  hccl::HcclAllToAllParams params;
  auto rank_size_imm = GetValue<int64_t>(rank_size);
  size_t count = SizeOf(input_tensor->shape()) / LongToSize(rank_size_imm);
  params.sendcount = count;
  params.recvcount = count;
  auto hccl_data_type = HcomUtil::ConvertHcclType(input_tensor->data_type());
  const auto &op_name = op->primitive()->name();
  auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
  auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
  auto launch_func = [input_data_ptr, output_data_ptr, hccl_data_type, params](const HcclComm &hccl_comm,
                                                                               void *comm_stream_ptr) {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllToAll(input_data_ptr, output_data_ptr, params,
                                                                     hccl_data_type, comm_stream_ptr, hccl_comm);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(EXCEPTION) << "HcclAllToAll failed, ret:" << hccl_result;
    }
  };
  return launch_func;
}
}  // namespace

void DistCommAllToAllVSingleAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                            const TensorPtr &input_tensor, const StringImmPtr &group,
                                            const ValueTuplePtr &send_numel_list, const ValueTuplePtr &recv_numel_list,
                                            const Int64ImmPtr &rank_size, const BoolImmPtr &split_sizes_empty) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, group, send_numel_list, recv_numel_list, rank_size,
                          split_sizes_empty);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  // PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, group, send_numel_list, recv_numel_list, rank_size,
                   split_sizes_empty]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, {other_tensor});

    auto is_split_sizes_empty = GetValue<bool>(split_sizes_empty);
    // Call AlltoAll for better performance when split_sizes is empty.
    const auto &launch_func = is_split_sizes_empty ? CallAllToAll(op, input_tensor, rank_size)
                                                   : CallAllToAllV(op, input_tensor, send_numel_list, recv_numel_list);

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
