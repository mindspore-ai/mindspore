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

#include "kernel/ascend/hccl/pyboost/dist_comm_all_to_all_v.h"

#include <memory>
#include <string>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

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

std::function<void(const HcclComm &, void *)> CallAllToAllVList(const std::shared_ptr<OpRunner> &op,
                                                                const std::vector<TensorPtr> &output_tensors,
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
  const auto &recv_numel_list_vec = ConvertValueTupleToVector<int64_t>(recv_numel_list);
  auto launch_func = [op_name, output_tensors, recv_numel_list_vec, input_data_ptr, output_data_ptr, hccl_data_type,
                      params](const HcclComm &hccl_comm, void *comm_stream_ptr) {
    auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAlltoAllV(input_data_ptr, output_data_ptr, params,
                                                                      hccl_data_type, comm_stream_ptr, hccl_comm);
    if (hccl_result != HCCL_SUCCESS) {
      MS_LOG(EXCEPTION) << "HcclAllToAllv failed, ret:" << hccl_result;
    }

    uint64_t offset = 0;
    for (size_t i = 0; i < output_tensors.size(); i++) {
      auto size = output_tensors[i]->Size();
      if (size == 0) {
        continue;
      }
      auto data_ptr = GetDevicePtrFromTensor(op_name, output_tensors[i]);
      auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, data_ptr, size, static_cast<char *>(output_data_ptr) + offset,
                                    size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
      if (cp_ret != EOK) {
        MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
      }
      offset += size;
    }
  };
  return launch_func;
}

}  // namespace

void DistCommAllToAllVAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &outputs,
                                      const TensorPtr &input_tensor, const StringImmPtr &group,
                                      const ValueTuplePtr &send_numel_list, const ValueTuplePtr &recv_numel_list,
                                      const Int64ImmPtr &rank_size) {
  OpRunner::InferOpOutput(op, outputs, input_tensor, group, send_numel_list, recv_numel_list, rank_size);

  std::vector<TensorPtr> output_tensors = ConvertValueTupleToVector<TensorPtr>(outputs);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, output_tensors, input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, op->outputs());

  auto run_func = [op, output_tensors, input_tensor, group, send_numel_list, recv_numel_list, rank_size]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, {output_tensors});
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, op->outputs());

    // Call AlltoAll for better performance when split_sizes is empty.
    const auto &launch_func = CallAllToAllVList(op, output_tensors, input_tensor, send_numel_list, recv_numel_list);

    auto post_func = [input_tensor, output_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor,
                                                                      op->output(0), output_tensors);
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
