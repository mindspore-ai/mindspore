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

#include "kernel/ascend/hccl/pyboost/dist_comm_reduce_scatter_tensor_uneven.h"

#include <memory>
#include <string>
#include <vector>
#include <numeric>
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
inline void FillEqualSplitSizes(std::vector<int64_t> *split_sizes, int64_t rank_size, int64_t input_tensor_size) {
  if (split_sizes->size() == 0) {
    // fill equal split sizes when empty
    int64_t split_val = input_tensor_size / rank_size;
    for (int i = 0; i < rank_size; i++) {
      split_sizes->push_back(split_val);
    }
  }
}

inline void CheckSplitSizes(const std::vector<int64_t> &split_sizes, int64_t rank_size, int64_t input_size) {
  if (split_sizes.size() != LongToSize(rank_size)) {
    MS_EXCEPTION(ValueError) << "For Primitive ReduceScatterTensorUneven"
                             << ", the num of input_split_sizes must be equal to rank_size, but got "
                             << split_sizes.size() << " and " << rank_size << ".";
  }
  int64_t sum_val = std::accumulate(split_sizes.begin(), split_sizes.end(), 0LL);
  if (sum_val != input_size) {
    MS_EXCEPTION(ValueError) << "For Primitive ReduceScatterTensorUneven "
                             << ", the sum of input_split_sizes must be equal to input_tensor_size, but got " << sum_val
                             << " and " << input_size << ".";
  }
}
}  // namespace

void DistCommReduceScatterTensorUnevenAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const TensorPtr &other_tensor, const TensorPtr &input_tensor,
                                                      const ValueTuplePtr &input_split_sizes,
                                                      const Int64ImmPtr &rank_size, const StringImmPtr &op_type,
                                                      const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, input_split_sizes, rank_size, op_type, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, input_split_sizes, rank_size, op_type, group]() {
    PyBoostUtils::MallocOpInputsWithStream(op->device_context(), kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, {other_tensor});
    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);
    auto op_type_enum = HcomUtil::GetHcomReduceOpType(GetValue<std::string>(op_type));

    auto input_split_sizes_vec = ConvertValueTupleToVector<int64_t>(input_split_sizes);
    auto rank_size_val = GetValue<int64_t>(rank_size);
    int64_t input_size = input_tensor->DataSize();
    auto input_shape = input_tensor->shape();
    int64_t input_dim0 = input_shape[kIndex0];
    int64_t input_dim0_stride = input_size / input_dim0;
    FillEqualSplitSizes(&input_split_sizes_vec, rank_size_val, input_dim0);
    CheckSplitSizes(input_split_sizes_vec, rank_size_val, input_dim0);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto other_data_ptr = GetDevicePtrFromTensor(op_name, other_tensor);
    auto recv_count = other_tensor->DataSize();
    auto other_tensor_size = other_tensor->Size();

    auto launch_func = [input_data_ptr, input_split_sizes_vec, recv_count, hccl_count, hccl_data_type,
                        other_tensor_size, other_data_ptr, input_dim0_stride,
                        op_type_enum](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      hccl::HcclReduceScatterVParams params_;
      const size_t rank_num = input_split_sizes_vec.size();
      params_.send_counts.reserve(rank_num);
      params_.sdispls.reserve(rank_num);
      uint64_t offset = 0;
      for (size_t i = 0; i < rank_num; i++) {
        auto count = LongToSize(input_split_sizes_vec[i]);
        params_.send_counts.push_back(count * input_dim0_stride);
        params_.sdispls.push_back(offset * input_dim0_stride);
        offset += count;
      }
      params_.recv_count = recv_count;

      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatterV(
        input_data_ptr, other_data_ptr, params_, hccl_data_type, op_type_enum, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclReduceScatterV failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
