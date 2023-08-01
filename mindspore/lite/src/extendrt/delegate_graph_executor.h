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
#ifndef MINDSPORE_LITE_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_H_
#include <vector>
#include <memory>
#include "include/api/delegate_api.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "runtime/hardware/device_context.h"
#include "tools/common/func_graph_subgraph.h"
#include "kernel/kernel.h"
namespace mindspore {
// Graph sink delegate, the whole FuncGraph as a node to execute.
class GraphSinkDelegate : public IDelegate<FuncGraph, CNode, kernel::KernelMod> {
 public:
  GraphSinkDelegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : IDelegate<FuncGraph, CNode, kernel::KernelMod>(inputs, outputs) {}
  virtual ~GraphSinkDelegate() = default;
  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) override;

  bool IsDelegateNode(const std::shared_ptr<CNode> &node) override;

 protected:
  FuncGraphPtr sink_graph_;
};

// wrap graph executor as delegate
class GraphExecutorDelegate : public GraphSinkDelegate {
 public:
  explicit GraphExecutorDelegate(const std::vector<mindspore::MSTensor> &inputs,
                                 const std::vector<mindspore::MSTensor> &outputs,
                                 std::shared_ptr<device::GraphExecutor> executor)
      : GraphSinkDelegate(inputs, outputs), executor_(executor) {}
  virtual ~GraphExecutorDelegate() = default;
  std::shared_ptr<kernel::KernelMod> CreateKernel(const std::shared_ptr<CNode> &node) override;

 private:
  const std::shared_ptr<device::GraphExecutor> executor_;
};
}  // namespace mindspore
#endif
