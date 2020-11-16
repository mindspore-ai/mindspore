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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"

namespace mindspore {
namespace opt {
using kernel::DumpOption;

constexpr auto kGraphKernelModule = "mindspore._extends.graph_kernel";
constexpr auto kGraphKernelSplitFunc = "split_with_json";
constexpr auto kGetGraphKernelOpExpander = "get_op_expander";
constexpr auto kJsonKeyMultiGraph = "multi_graph";
constexpr auto kJsonKeyGraphDesc = "graph_desc";
constexpr auto kJsonKeyGraphMode = "graph_mode";

bool ConvertNonscalarTensorToParameter(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr);
std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> MixedNodesTransToGraph(const AnfNodePtrList &fuse_nodes,
                                                                                AnfNodePtrList *src_outputs = nullptr);
void SetNewKernelInfo(const AnfNodePtr &new_node, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                      const AnfNodePtrList &outputs, kernel::Processor processor);
AnfNodePtrList GetExpandOuts(const AnfNodePtrList &outs);
AnfNodePtr CreateNewFuseCNode(const FuncGraphPtr &kernel_graph, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                              const AnfNodePtrList &outputs);
void ReplaceNewFuseCNode(const FuncGraphPtr &kernel_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs);
void FuseNodesToSubGraph(const std::vector<AnfNodePtr> &fuse_nodes,
                         const std::shared_ptr<session::KernelGraph> &kernel_graph, const std::string &postfix);
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc);
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc,
                   std::map<std::string, AnfNodePtr> *address_node_map);
bool AnfToJsonDesc(const std::vector<AnfNodePtrList> &graphs, const DumpOption &dump_option, nlohmann::json *op_desc);
FuncGraphPtr JsonDescToAnf(const std::string &json_desc, const std::vector<AnfNodePtr> &inputs);
std::unordered_set<PrimitivePtr> GetExpandOps();
std::string ExtractGraphKernelName(const AnfNodePtrList &cnodes, const string &prefix = "", const string &postfix = "");
std::vector<PrimitivePtr> GetFusibleOpList();
bool IsBasicFuseOp(const AnfNodePtr &node);
void ResetKernelInfo(const AnfNodePtr &node, KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
