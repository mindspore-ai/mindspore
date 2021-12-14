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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <queue>
#include <set>
#include <unordered_map>
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "src/inner_context.h"
#include "src/tensor.h"
#include "src/sub_graph_kernel.h"
#include "include/model.h"

namespace mindspore::lite {
class ControlFlowScheduler {
 public:
  ControlFlowScheduler(InnerContext *ctx, const mindspore::Context *ms_ctx, std::vector<Tensor *> *src_tensors)
      : context_(ctx), ms_context_(ms_ctx), src_tensors_(src_tensors) {}
  ~ControlFlowScheduler() = default;
  int SplitNonTailCallSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernels);
  void RecordPartialNodeCallMoreThanOnce(kernel::LiteKernel *partial_node);
  // we insert entrance subgraph kernel and exit subgraph kernel define the boundary of the subgraph.
  int BuildBoundaryForMultipleCalledGraph(std::vector<kernel::LiteKernel *> *dst_kernels);

 private:
  bool IsNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel);
  int SplitSingleNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel,
                                     std::vector<kernel::LiteKernel *> *subgraph_kernels);
  std::set<kernel::LiteKernel *> GetNonTailCallSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernels);
  void RemoveUselessKernels(std::vector<kernel::LiteKernel *> *dst_kernels,
                            const std::set<kernel::LiteKernel *> &useless_kernels);
  void AppendToProcessQ(std::vector<kernel::LiteKernel *> *new_subgraphs,
                        std::set<kernel::LiteKernel *> *all_non_tail_subgraphs);
  // link partial output to call output.
  int RecordNonTailCallLinkInfo();
  kernel::SubGraphKernel *CreateEntranceSubGraph(kernel::SubGraphKernel *subgraph, lite::Tensor *link_tensor);
  kernel::SubGraphKernel *CreateExitSubGraph(kernel::SubGraphKernel *subgraph, lite::Tensor *link_tensor);

 private:
  InnerContext *context_ = nullptr;
  const mindspore::Context *ms_context_ = nullptr;
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
  std::vector<Tensor *> *src_tensors_ = nullptr;
  std::queue<kernel::LiteKernel *> to_process_q_{};
  std::vector<kernel::LiteKernel *> non_tail_calls_{};
  // key is partial node, value is the corresponding call node.
  std::set<kernel::LiteKernel *> more_than_once_called_partial_nodes_{};
  // record subgraph which has been inserted entrance and exit subgraph node, the key is subgraph kernel, the value is
  // the exit kernel.
  std::unordered_map<kernel::LiteKernel *, kernel::LiteKernel *> subgraph_kernel_and_exit_kernel_{};
};

using ControlFlowSchedulerPtr = std::shared_ptr<ControlFlowScheduler>;
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
