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

#include "kernel/ascend/hccl/pyboost/dist_comm_reduce_scatter.h"

#include <memory>
#include <string>
#include "ir/tensor_new.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/pyboost/comm_common.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommReduceScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                          const ValueTuplePtr &input_list, const Int64ImmPtr &rank_size,
                                          const StringImmPtr &op_type, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_list, rank_size, op_type, group);

  std::vector<TensorPtr> scatter_tensors = ConvertValueTupleToVector<TensorPtr>(input_list);
  op->set_outputs({other_tensor});

  auto rank_size_imm = GetValue<int64_t>(rank_size);
  auto input_shape = other_tensor->shape();
  int64_t input_shape_size = 0;
  for (const auto &scatter_tensor : scatter_tensors) {
    auto shape = scatter_tensor->shape();
    input_shape_size += static_cast<int64_t>(shape[0]);
  }
  input_shape[0] = static_cast<int64_t>(input_shape_size);
  TensorPtr input_tensor =
    tensor::from_spec(static_cast<TypeId>(other_tensor->data_type_c()), input_shape, device::DeviceType::kNone);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, scatter_tensors);
  // Avoid h2d copy.
  PyBoostUtils::PrepareOpOutputs(op->device_context(), kDefaultStreamIndex, {input_tensor});

  auto run_func = [op, other_tensor, input_tensor, rank_size_imm, op_type, group, scatter_tensors]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputsWithStream(device_context, kDefaultStreamIndex, scatter_tensors);
    PyBoostUtils::MallocOpOutputsWithStream(device_context, kDefaultStreamIndex, {other_tensor, input_tensor});
    auto hccl_data_type = HcomUtil::ConvertHcclType(other_tensor->data_type());
    auto hccl_count = 1;
    auto out_shape = other_tensor->shape();
    for (size_t i = 0; i < out_shape.size(); i++) {
      hccl_count = hccl_count * out_shape[i];
    }

    auto op_type_enum = HcomUtil::GetHcomReduceOpType(GetValue<std::string>(op_type));

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto output_data_ptr = GetDevicePtrFromTensor(op_name, op->output(0));
    auto size = scatter_tensors[0]->Size();
    auto other_tensor_num_elements = other_tensor->DataSize();
    auto launch_func = [input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum, size, rank_size_imm,
                        scatter_tensors, op_name,
                        other_tensor_num_elements](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      bool same_shape = std::all_of(scatter_tensors.begin(), scatter_tensors.end(),
                                    [&](const TensorPtr &t) { return t->shape() == scatter_tensors[0]->shape(); });
      if (same_shape) {
        for (int r = 0; r < rank_size_imm; r++) {
          uint64_t offset = static_cast<uint64_t>(r * size);
          auto data_ptr = GetDevicePtrFromTensor(op_name, scatter_tensors[r]);
          auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, static_cast<char *>(input_data_ptr) + offset, size, data_ptr,
                                        size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
          }
        }
        auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatter(
          input_data_ptr, output_data_ptr, hccl_count, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
        if (hccl_result != HCCL_SUCCESS) {
          MS_LOG(EXCEPTION) << "HcclReduceScatter failed, ret:" << hccl_result;
        }
      } else {
        MS_LOG(DEBUG) << "For kernel HcclReduceScatter, Different shapes detected, using HcclReduceScatterV instead.";
        hccl::HcclReduceScatterVParams params;
        params.send_counts.clear();
        params.sdispls.resize(rank_size_imm, 0);

        for (const auto &t : scatter_tensors) {
          params.send_counts.push_back(t->DataSize());
        }
        for (int r = 1; r < rank_size_imm; ++r) {
          params.sdispls[r] = params.sdispls[r - 1] + params.send_counts[r - 1];
        }
        params.recv_count = other_tensor_num_elements;

        uint64_t offset = 0;
        for (int r = 0; r < rank_size_imm; r++) {
          auto data_ptr = GetDevicePtrFromTensor(op_name, scatter_tensors[r]);
          uint64_t send_size = scatter_tensors[r]->Size();
          auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, static_cast<char *>(input_data_ptr) + offset, send_size,
                                        data_ptr, send_size, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream_ptr);
          if (cp_ret != EOK) {
            MS_LOG(EXCEPTION) << "aclrtMemcpy failed.";
          }
          offset += send_size;
        }

        auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatterV(
          input_data_ptr, output_data_ptr, params, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
        if (hccl_result != HCCL_SUCCESS) {
          MS_LOG(EXCEPTION) << "HcclReduceScatterV failed, ret:" << hccl_result;
        }
      }
    };

    auto post_func = [input_tensor, scatter_tensors, op](const DeviceEventPtr &event, size_t comm_stream_id) {
      runtime::DeviceAddressUtils::ProcessCrossStreamAddressWithEvent(op->primitive()->name(), op->device_context(),
                                                                      comm_stream_id, event, input_tensor,
                                                                      op->output(0), scatter_tensors);
    };
    CommonCommAscendFunc(op, input_tensor, group, launch_func, post_func);
  };
  CommonCommRunTask(run_func);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
