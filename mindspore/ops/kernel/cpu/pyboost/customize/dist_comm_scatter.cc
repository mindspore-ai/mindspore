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

#include "mindspore/ops/kernel/cpu/pyboost/customize/dist_comm_scatter.h"
#include <memory>
#include <utility>
#include <string>
#include "ir/tensor_new.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "utils/misc.h"
#endif

namespace mindspore {
namespace kernel {
namespace pyboost {
void DistCommScatterCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &other_tensor,
                                 const ValueTuplePtr &scatter_list, const Int64ImmPtr &rank_size,
                                 const Int64ImmPtr &src, const Int64ImmPtr &rank_id, const StringImmPtr &group) {
#if defined(__linux__) && defined(WITH_BACKEND)
  OpRunner::InferOpOutput(op, other_tensor, scatter_list, rank_size, src, rank_id, group);
  op->set_outputs({other_tensor});

  auto src_rank = GetValue<int64_t>(src);
  auto local_rank = GetValue<int64_t>(rank_id);
  std::vector<TensorPtr> scatter_tensors = ConvertValueTupleToVector<TensorPtr>(scatter_list);

  auto rank_size_imm = static_cast<size_t>(GetValue<int64_t>(rank_size));
  auto input_shape = other_tensor->shape();
  input_shape[0] = static_cast<int64_t>(input_shape[0] * rank_size_imm);
  TensorPtr input_tensor =
    tensor::from_spec(static_cast<TypeId>(other_tensor->data_type_c()), input_shape, device::DeviceType::kCPU);
  PyBoostUtils::PrepareOpInputs(op->device_context(), kDefaultStreamIndex, other_tensor, scatter_tensors);

  auto run_func = [op, other_tensor, local_rank, src_rank, group, rank_size_imm, scatter_tensors, input_tensor]() {
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputs(device_context, scatter_tensors);
    PyBoostUtils::MallocOpOutputs(device_context, {other_tensor});

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), input_tensor);
    auto in_addr = input_address_info.first;
    const auto &other_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), other_tensor);
    auto other_addr = other_address_info.first;
    auto out_size = other_addr[0]->size();
    if (local_rank == src_rank) {
      const auto &scatter_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), scatter_tensors);
      for (size_t r = 0; r < rank_size_imm; r++) {
        auto scatter_addr = (scatter_address_info.first)[r]->device_ptr();
        auto input_addr = in_addr[0]->device_ptr();
        size_t offset = static_cast<size_t>(r * out_size);
        auto mem_ret = Memcpy(reinterpret_cast<char *>(input_addr) + offset, out_size,
                              reinterpret_cast<char *>(scatter_addr), out_size);
        if (mem_ret != EOK) {
          MS_LOG(EXCEPTION) << "Memcpy failed. ret is " << mem_ret;
        }
      }
    }
    const auto &group_str = GetValue<std::string>(group);
    size_t type_len = GetDataTypeSize(in_addr[0]->dtype_id());

    auto comm_lib = distributed::collective::CollectiveManager::instance()->host_comm_lib();
    bool ret = comm_lib->Scatter(in_addr[0]->device_ptr(), other_addr[0]->device_ptr(), out_size / type_len,
                                 in_addr[0]->dtype_id(), src_rank, group_str);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Scatter failed.";
    }
  };
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(run_func));
#else
  MS_LOG(EXCEPTION) << "The CPU op scatter is only supported on linux platform.";
#endif
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
