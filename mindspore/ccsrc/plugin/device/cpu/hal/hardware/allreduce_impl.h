/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_ALLREDUCE_IMPL_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_ALLREDUCE_IMPL_H_

#include "distributed/cluster/cluster_context.h"

namespace mindspore {
namespace device {
namespace cpu {
class AllReduceLauncher {
 public:
  AllReduceLauncher(const AllReduceLauncher &) = delete;
  AllReduceLauncher &operator=(const AllReduceLauncher &) = delete;
  ~AllReduceLauncher() = default;

  static AllReduceLauncher &GetInstance() {
    static AllReduceLauncher instance;
    return instance;
  }
  bool Execute(const void *input_data, void *const output_data, size_t data_size) const;

 private:
  size_t rank_id_{0};
  size_t rank_size_{0};
  ps::core::NodeRole node_role_{ps::core::WORKER};
  ps::core::AbstractNodePtr abs_node_{nullptr};

  AllReduceLauncher();

  bool RingAllReduce(const void *input_data, void *const output_data, size_t data_size) const;
  bool ReduceBroadcastAllReduce(const void *input_data, void *const output_data, size_t data_size) const;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_ALLREDUCE_IMPL_H_
