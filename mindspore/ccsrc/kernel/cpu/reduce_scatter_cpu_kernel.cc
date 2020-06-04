/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/reduce_scatter_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "device/cpu/mpi/mpi_adapter.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kRanksGroup = "group";
}  // namespace

ReduceScatterCPUKernel::ReduceScatterCPUKernel() : output_data_number_(0), op_type_(device::cpu::kOpTypeSum) {}

void ReduceScatterCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t i = 0; i < output_num; ++i) {
    auto shape = AnfAlgo::GetOutputInferShape(kernel_node, i);
    size_t size = 1;
    for (size_t j = 0; j < shape.size(); j++) {
      size *= IntToSize(shape[j]);
    }
    output_data_number_ += size;
  }
  auto op = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("op");
  if (op != nullptr) {
    op_type_ = GetValue<std::string>(op);
  }

  auto ranks_group = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kRanksGroup);
  if (ranks_group != nullptr) {
    ranks_group_ = GetValue<std::vector<int>>(ranks_group);
  } else {
    MS_LOG(EXCEPTION) << "Miss attribute " << kRanksGroup;
  }
}

bool ReduceScatterCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> & /*workspace*/,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  return device::cpu::MPIAdapter::Instance().ReduceScatter(input_addr, output_addr, ranks_group_, output_data_number_,
                                                           op_type_);
}
}  // namespace kernel
}  // namespace mindspore
