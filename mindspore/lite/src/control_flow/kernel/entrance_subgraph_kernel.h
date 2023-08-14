/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_ENTRANCE_SUBGRAPH_KERNEL_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_ENTRANCE_SUBGRAPH_KERNEL_H_
#include <atomic>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/executor/kernel_exec.h"
#include "src/litert/executor.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/litert/cpu_info.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore::kernel {
class EntranceSubGraphKernel : public SubGraphKernel {
 public:
  explicit EntranceSubGraphKernel(MSKernel *kernel) : SubGraphKernel({}, {}, {}, kernel) {
    subgraph_type_ = kEntranceSubGraph;
  }

  ~EntranceSubGraphKernel() override = default;

  static SubGraphKernel *Create(MSKernel *kernel);

  int Prepare() override { return RET_OK; };

  int Execute() override { return Execute(nullptr, nullptr); }

  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;

  int ReSize() override { return RET_OK; };
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_KERNEL_ENTRANCE_SUBGRAPH_KERNEL_H_
