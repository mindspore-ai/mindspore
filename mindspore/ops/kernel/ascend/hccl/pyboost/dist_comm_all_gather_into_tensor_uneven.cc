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

#include "kernel/ascend/hccl/pyboost/dist_comm_all_gather_into_tensor_uneven.h"

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
inline void FillEqualSplitSizes(std::vector<int64_t> *split_sizes, int64_t rank_size, int64_t other_tensor_size) {
  if (split_sizes->size() == 0) {
    // fill equal split sizes when empty
    int64_t split_val = other_tensor_size / rank_size;
    for (int i = 0; i < rank_size; i++) {
      split_sizes->push_back(split_val);
    }
  }
}

inline void CheckSplitSizes(const std::vector<int64_t> &split_sizes, int64_t rank_size, int64_t output_size) {
  if (split_sizes.size() != LongToSize(rank_size)) {
    MS_EXCEPTION(ValueError) << "For Primitive AllGatherIntoTensorUneven"
                             << ", the num of output_split_sizes must be equal to rank_size, but got "
                             << split_sizes.size() << " and " << rank_size << ".";
  }
  int64_t sum_val = std::accumulate(split_sizes.begin(), split_sizes.end(), 0LL);
  if (sum_val != output_size) {
    MS_EXCEPTION(ValueError) << "For Primitive AllGatherIntoTensorUneven "
                             << ", the sum of output_split_sizes must be equal to other_tensor_size, but got "
                             << sum_val << " and " << output_size << ".";
  }
}
}  // namespace

void DistCommAllGatherIntoTensorUnevenAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const TensorPtr &other_tensor, const TensorPtr &input_tensor,
                                                      const ValueTuplePtr &output_split_sizes,
                                                      const Int64ImmPtr &rank_size, const StringImmPtr &group) {
  OpRunner::InferOpOutput(op, other_tensor, input_tensor, output_split_sizes, rank_size, group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, input_tensor);
  op->set_outputs({other_tensor});

  auto run_func = [op, other_tensor, input_tensor, output_split_sizes, rank_size, group]() {
    PyBoostUtils::MallocOpInputsWithStream(op->device_context(), kDefaultStreamIndex, input_tensor);
    PyBoostUtils::MallocOpOutputsWithStream(op->device_context(), kDefaultStreamIndex, {other_tensor});
    auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(op->primitive(), input_tensor);

    auto output_split_sizes_vec = ConvertValueTupleToVector<int64_t>(output_split_sizes);
    auto rank_size_val = GetValue<int64_t>(rank_size);
    int64_t output_size = other_tensor->DataSize();
    auto output_shape = other_tensor->shape();
    int64_t output_dim0 = output_shape[kIndex0];
    int64_t output_dim0_stride = output_size / output_dim0;
    FillEqualSplitSizes(&output_split_sizes_vec, rank_size_val, output_dim0);
    CheckSplitSizes(output_split_sizes_vec, rank_size_val, output_dim0);

    const auto &op_name = op->primitive()->name();
    auto input_data_ptr = GetDevicePtrFromTensor(op_name, input_tensor);
    auto other_data_ptr = GetDevicePtrFromTensor(op_name, other_tensor);
    auto send_count = input_tensor->DataSize();

    auto launch_func = [input_data_ptr, output_split_sizes_vec, send_count, hccl_count, hccl_data_type, other_data_ptr,
                        output_dim0_stride](const HcclComm &hccl_comm, void *comm_stream_ptr) {
      hccl::HcclAllGatherVParams params_;
      params_.send_count = send_count;

      const size_t rank_num = output_split_sizes_vec.size();
      params_.recv_counts.reserve(rank_num);
      params_.rdispls.reserve(rank_num);

      uint64_t offset = 0;
      for (size_t i = 0; i < rank_num; ++i) {
        const auto count = LongToSize(output_split_sizes_vec[i]);
        params_.recv_counts.emplace_back(count * output_dim0_stride);
        params_.rdispls.emplace_back(offset * output_dim0_stride);
        offset += count;
      }
      auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGatherV(input_data_ptr, other_data_ptr, params_,
                                                                         hccl_data_type, comm_stream_ptr, hccl_comm);
      if (hccl_result != HCCL_SUCCESS) {
        MS_LOG(EXCEPTION) << "HcclAllGatherV failed, ret:" << hccl_result;
      }
    };

    CommonCommAscendFunc(op, input_tensor, group, launch_func, nullptr);
  };
  CommonCommRunTask(run_func);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
