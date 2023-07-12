/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXECUTOR_SUB_GRAPH_KERNEL_DRAWER_H_
#define MINDSPORE_LITE_SRC_EXECUTOR_SUB_GRAPH_KERNEL_DRAWER_H_

#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "src/common/graphviz_drawer.h"
#include "include/errorcode.h"
#include "src/executor/kernel_exec.h"
#include "src/executor/sub_graph_kernel.h"
#include "src/executor/drawer_mark_filter.h"

namespace mindspore::lite {
class SubGraphKernelGVGraph : public GVGraph {
 public:
  static std::shared_ptr<SubGraphKernelGVGraph> Create(const kernel::SubGraphKernel &sub_graph,
                                                       const MarkFilter &mark_filter);

  static std::shared_ptr<SubGraphKernelGVGraph> Create(const kernel::SubGraphKernel &sub_graph,
                                                       const std::vector<schema::PrimitiveType> &mark_types);

  explicit SubGraphKernelGVGraph(const std::string &name) : GVGraph(name) {}
  void AppendGraphInputNode(const lite::Tensor &tensor);
  int AppendWeightNode(const lite::Tensor &tensor, const std::string &name);
  int AppendKernelExecNode(const kernel::KernelExec &kernel, bool highlight = false);
  int AppendGraphOutputNode(const std::vector<lite::Tensor *> &out_tensors);

 protected:
  static GVNode *CreateKernelExecNode(const kernel::KernelExec &kernel, bool highlight = false);
  int LinkNodes(const kernel::KernelExec &kernel, const GVNode &gv_node);
  void AppendOutTensorMap(const lite::Tensor *tensor, lite::GVNode *node, size_t out_index);
  std::pair<lite::GVNode *, size_t> GetBelongingGVNode(const lite::Tensor *tensor) const;

  std::string name_;
  std::unordered_map<const lite::Tensor *, std::pair<lite::GVNode *, size_t>> gv_node_out_tensor_map_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_EXECUTOR_SUB_GRAPH_KERNEL_DRAWER_H_
