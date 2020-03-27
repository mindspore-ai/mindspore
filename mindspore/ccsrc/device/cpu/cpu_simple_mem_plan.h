/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEVICE_CPU_CPU_SIMPLE_MEM_PLAN_H_
#define MINDSPORE_CCSRC_DEVICE_CPU_CPU_SIMPLE_MEM_PLAN_H_

#include <vector>
#include <unordered_map>
#include "session/kernel_graph.h"
#include "device/device_address.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan {
 public:
  CPUSimpleMemPlan() = default;
  ~CPUSimpleMemPlan() = default;

  void MemPlan(const session::KernelGraph *graph);
  void MemAssign(const session::KernelGraph *graph, uint8_t *base_ptr);
  size_t GetGraphMemSize(const session::KernelGraph *graph);

 private:
  std::unordered_map<const session::KernelGraph *, size_t> graph_mem_size_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_CPU_CPU_SIMPLE_MEM_PLAN_H_
