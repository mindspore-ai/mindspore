/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_GRAPH_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_GRAPH_UTILS_H_

#include <memory>
#include <utility>
#include <vector>

#include "backend/common/session/kernel_graph.h"

namespace mindspore::infer {
using GraphId = uint32_t;
class KernelGraphUtils {
 public:
  KernelGraphUtils() = default;
  virtual ~KernelGraphUtils() = default;

  static KernelGraphUtils &Instance() {
    static KernelGraphUtils instance;
    return instance;
  }

  KernelGraphPtr ConstructKernelGraph(const FuncGraphPtr &func_graph, std::vector<KernelGraphPtr> *all_out_graph,
                                      DeviceType device_target);
  KernelGraphPtr NewKernelGraph();
  ParameterPtr CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph);
  ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph);
  ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph);
  ParamInfoPtr GetParamDefaultValue(const AnfNodePtr &node);
  void InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter);
  AnfNodePtr CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph);
  ValueNodePtr CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph);
  bool CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph);
  void AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph);
  void SetInputNodeUsage(const KernelGraphPtr &graph, const FuncGraphManagerPtr &manager);
  GraphId GetGraphIdByNode(const AnfNodePtr &front_anf) const;
  KernelGraphPtr GetGraph(mindspore::GraphId graph_id) const;
  AnfNodePtr CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph);
  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph);
  void SetReturnNode(const AnfNodePtr &node, KernelGraph *graph);
  bool IsUsedByRealKernel(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, const uint32_t graph_id);
  bool IsShapeDynamic(const abstract::ShapePtr &shape);
  std::vector<AnfNodePtr> CreateValueNode(const CNodePtr &cnode, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph);
  void CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs);
  bool RecursiveCheck(const FuncGraphManagerPtr &manager, const std::pair<AnfNodePtr, int64_t> &kernel, size_t *idx);
  std::vector<AnfNodePtr> CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph);
  CNodePtr CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph);
  void ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph, const std::vector<AnfNodePtr> &real_inputs);

#ifndef ENABLE_SECURITY
  static bool KernelGraphUtils::ExistSummaryNode(const KernelGraph *graph);
#endif

 private:
  mindspore::HashMap<FuncGraph *, KernelGraphPtr> front_backend_graph_map_;
  static GraphId graph_sum_;
};
}  // namespace mindspore::infer

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_GRAPH_UTILS_H_
