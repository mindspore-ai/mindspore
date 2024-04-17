/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_MGR_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_MGR_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include "utils/hash_map.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/anfalgo.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/backend/kernel_info.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
using GraphId = uint32_t;
using GraphInfo = std::string;

namespace session {
#ifndef ENABLE_SECURITY
bool ExistSummaryNode(const KernelGraph *graph);
#endif
ParamInfoPtr GetParamDefaultValue(const AnfNodePtr &node);

struct PartialFuncInfo {
  AbstractBasePtr abstract;
  AnfNodePtr sub_graph;
  size_t param_begin;
  size_t param_end;
  size_t multi_tuple;
};

class BACKEND_EXPORT KernelGraphMgr {
 public:
  KernelGraphMgr() {}
  virtual ~KernelGraphMgr() {}

  // The parameter is_enable_zero_copy means if the parameter in graph can avoid copy when it is executed, and it is
  // true in subgraph sink mode, and the device address shared for partial parameters and internal parameters in graph
  // would be disabled.
  std::shared_ptr<KernelGraph> ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs,
                                                    DeviceType device_target = DeviceType::kUnknown,
                                                    bool common_opt = true, bool is_enable_zero_copy = false);

  std::shared_ptr<KernelGraph> ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                    std::vector<KernelGraphPtr> *all_out_graph,
                                                    DeviceType device_target);

  std::shared_ptr<KernelGraph> ConstructKernelGraph(std::vector<KernelGraphPtr> *all_out_graph);
  std::shared_ptr<KernelGraph> ConstructPackKernelGraph(const FuncGraphPtr &func_graph,
                                                        std::vector<KernelGraphPtr> *all_out_graph,
                                                        DeviceType device_target);

  void SetInputNodeUsage(const KernelGraphPtr &graph, const FuncGraphManagerPtr &manager) const;

  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                          mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode);

  // get graph id in child graphs by ME front anf node pointer
  virtual GraphId GetGraphIdByNode(const AnfNodePtr &) const;

  // Get graph by graph id, if not exist return null ptr
  KernelGraphPtr GetGraph(GraphId graph_id) const;
  void ClearGraph();
  virtual void UnifyMindIR(const KernelGraphPtr &graph);
  virtual ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph);
  // create a new kernel graph and update the graph sum
  KernelGraphPtr NewKernelGraph();
  KernelGraphPtr NewPynativeKernelGraph();
  void SetKernelGraphId(const KernelGraphPtr &kernel_graph);
  AnfNodePtr CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) const;

  AnfNodePtr CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph);
  ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) const;
  bool CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph);
  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph);

  GraphId GraphSum() const { return graph_sum_; }
  void ClearPartialParameterMap() { partial_parameters_map_.clear(); }

  mindspore::HashMap<FuncGraph *, KernelGraphPtr> GetFrontBackendGraphMap() const { return front_backend_graph_map_; }
  void CacheKernelGraph(const KernelGraphPtr &kg);
  // do inline
  static AnfNodePtr DoInline(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                             const AnfNodePtrList &func_graph_args, const ScopePtr &scope,
                             const uint32_t &target_graph_id,
                             const std::map<session::AnfWithOutIndex, session::AnfWithOutIndex> &ref_map,
                             const KernelGraphPtr &graph, bool is_switch_inline);

 private:
  void GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) const;
  void GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                         mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode);
  AnfNodePtr GetChildGraph(KernelGraph *graph, const AnfNodePtr &child_func_graph);
  void HandleInternalOutput(const AnfNodePtr &input_front_node, const AnfNodePtr &backend_node,
                            const FuncGraphManagerPtr &front_func_graph_manager,
                            const std::shared_ptr<KernelGraph> &backend_graph);
  std::string AddPartialParametersMap(const AnfNodePtr &partial_node);

  CNodePtr CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateValueNode(const CNodePtr &cnode, KernelGraph *graph);
  void CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs);
  std::vector<AnfNodePtr> CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) const;

  std::vector<AnfNodePtr> CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph);
  void ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph, const std::vector<AnfNodePtr> &real_inputs);

  ValueNodePtr CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph);
  ParameterPtr CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) const;
  void AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) const;
  void SetReturnNode(const AnfNodePtr &node, KernelGraph *graph);
  void FlattenTuple(const CNodePtr &node);
  bool ParseKernelGraphNodesAndAttrs(const nlohmann::json &model_json);

 protected:
  CNodePtr ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph);

  void InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) const;
  void ConstructKernelGraphInner(const FuncGraphPtr &func_graph, std::vector<KernelGraphPtr> *all_out_graph,
                                 DeviceType device_target, const KernelGraphPtr &graph);

  mindspore::HashMap<GraphId, std::shared_ptr<KernelGraph>> graphs_;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> partial_parameters_map_;
  mindspore::HashMap<AnfNodePtr, std::string> partial_target_map_;
  mindspore::HashMap<AnfNodePtr, ParameterPtr> default_param_map_;
  mindspore::HashMap<FuncGraph *, KernelGraphPtr> front_backend_graph_map_;
  mindspore::HashMap<KernelGraph *, PartialFuncInfo> kernel_graph_partial_map_;
  mindspore::HashSet<AnfNodePtr> need_flatten_;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> need_flatten_tuple_map_;
  static GraphId graph_sum_;
  static GraphId pynative_graph_sum_;
  // record all graphs's backend params unique name to its weak_ptr
  // graph can come from different frontend graph
  static mindspore::HashMap<std::string, std::weak_ptr<AnfNode>> name_to_params_;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_MGR_H
