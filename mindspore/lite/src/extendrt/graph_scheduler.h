/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_H
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "src/litert/kernel_exec.h"

namespace mindspore {
namespace infer {
struct ScheduleStrategy {};
struct GraphCompilerInfo;

struct Tensor {};
struct ExcutionPlan {
  std::vector<kernel::KernelExec> kernels_;
  std::vector<Tensor> tensors_;
  std::vector<int64_t> inputs_;
  std::vector<int64_t> outputs_;
};

class GraphScheduler : public std::enable_shared_from_this<GraphScheduler> {
 public:
  explicit GraphScheduler(const ScheduleStrategy &strategy);
  virtual ~GraphScheduler();
  ExcutionPlan Schedule(const CompileResult &);
};
}  // namespace infer
}  // namespace mindspore
#endif
