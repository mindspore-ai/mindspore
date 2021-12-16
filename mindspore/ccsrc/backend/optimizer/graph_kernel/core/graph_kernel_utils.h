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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_

#include <string>
#include <tuple>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "backend/optimizer/graph_kernel/model/lite_graph.h"

namespace mindspore::graphkernel {
constexpr auto kGraphKernelDumpPath = "graph_kernel_dump";
constexpr auto kAllTarget = "ALL";

using OpWithLevel = std::tuple<std::string, unsigned int, PrimitivePtr>;

class GkUtils {
 public:
  /**
   * @brief Extract kernel name from nodes, only the real kernel CNode is processed.
   * @param[in] nodes The node list
   * @param[in] prefix The prefix of result name
   * @param[in] postfix The postfix of result name
   * @return The string concatenated by the names of all cnodes
   */
  static std::string ExtractGraphKernelName(const AnfNodePtrList &nodes, const std::string &prefix = "",
                                            const std::string &postfix = "");

  /**
   * @brief Spread the MakeTuple in node list
   * @param[in] nodes
   * @param[in] begin_index
   * @example
   *   input
   *     nodes: [ a, b, MakeTuple[i, j], c, d, MakeTuple[x, MakeTuple[y, z]] ]
   *     begin_index: 1
   *   output
   *     [b, i, j, c, d, x, y, z]
   * @return std::vector<AnfNodePtr>
   */
  static AnfNodePtrList SpreadTuples(const AnfNodePtrList &nodes, size_t begin_index = 0);

  /**
   * @brief Filter operators by target, op level, and enable/disable flags.
   * @param[in] ops_with_level the default operator list
   * @param[in] level enabled op level
   * @param[in] enable_ops_only the "enable_xxx_ops_only" flag
   * @param[in] enable_ops the "enable_xxx_ops" flag
   * @param[in] disable_ops the "disable_xxx_ops" flag
   * @return Available primitive list
   */
  static std::vector<PrimitivePtr> GetValidOps(const std::vector<OpWithLevel> &ops_with_level, unsigned int level,
                                               const std::vector<std::string> &enable_ops_only,
                                               const std::vector<std::string> &enable_ops,
                                               const std::vector<std::string> &disable_ops);

  /**
   * @brief Check whether graphkernel supports the node
   */
  static bool IsKeepBasicNode(const AnfNodePtr &node);

  /**
   * @brief Create CNode.
   */
  static CNodePtr NewRealCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph,
                               const std::vector<inner::NodeBase> &out_info_list);

  /**
   * @brief Change lite graph to anf graph.
   */
  static FuncGraphPtr LiteGraph2AnfGraph(const inner::LiteGraphPtr &lite_graph);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_
