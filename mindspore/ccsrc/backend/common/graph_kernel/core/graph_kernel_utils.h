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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_

#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/hash_map.h"
#include "backend/common/graph_kernel/model/lite_graph.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::prim {
GVAR_DEF(PrimitivePtr, kPrimElemAny, std::make_shared<Primitive>("ElemAny"));
GVAR_DEF(PrimitivePtr, kPrimLayoutTransform, std::make_shared<Primitive>("LayoutTransform"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceOnnx, std::make_shared<Primitive>("StridedSliceOnnx"));
GVAR_DEF(PrimitivePtr, kPrimSolveTriangular, std::make_shared<Primitive>("SolveTriangular"));
GVAR_DEF(PrimitivePtr, kPrimLU, std::make_shared<Primitive>("LU"));
}  // namespace mindspore::prim

namespace mindspore::graphkernel {
constexpr auto kGraphKernelDumpPath = "graph_kernel_dump";
constexpr auto kAllTarget = "ALL";
constexpr auto kOutputsFormat = "outputs_format";

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
   * @brief Filter excluded operators which can not be enabled from operator list.
   * @param[in] ops the default operator list
   * @return The operator list after removing the excluded operators
   */
  static std::vector<PrimitivePtr> FilterExcludedOps(const std::vector<PrimitivePtr> &ops);

  /**
   * @brief Check whether graphkernel supports the node
   */
  static bool IsKeepBasicNode(const AnfNodePtr &node);

  /**
   * @brief Create CNode.
   */
  static CNodePtr NewRealCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph,
                               const std::vector<inner::NodeBase> &out_info_list, const CallbackPtr &cb);

  /**
   * @brief Transform  LiteGraph to FuncGraph.
   */
  static FuncGraphPtr LiteGraph2AnfGraph(const inner::LiteGraphPtr &lite_graph, const CallbackPtr &cb);

  /**
   * @brief Transform FuncGraph to LiteGraph
   * @param[in] func_graph the anf FuncGraph
   * @param[out] op_node_map store the relationship of inner node and anf node.
   */
  static inner::LiteGraphPtr AnfGraph2LiteGraph(const FuncGraphPtr &func_graph,
                                                HashMap<inner::NodePtr, AnfNodePtr> *op_node_map = nullptr);

  /**
   * @brief Get manager of func graph. If there is no manager, a new one will be created.
   */
  static FuncGraphManagerPtr GetFuncGraphManager(const FuncGraphPtr &func_graph);

  /**
   * @brief Update func graph manager. Do nothing if the manager is nullptr.
   */
  static void UpdateFuncGraphManager(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);

  /**
   * @brief Get op's prim by op name.
   */
  static PrimitivePtr GetOpsPrim(const std::string &name);

  /**
   * @brief Get the real kernel nodes from a FuncGraph, the parameters and output nodes are extracted if the input_list
   * and output_list is set.
   */
  static void GetValidKernelNodes(const FuncGraphPtr &func_graph, AnfNodePtrList *node_list,
                                  AnfNodePtrList *input_list = nullptr, AnfNodePtrList *output_list = nullptr);

  /**
   * @brief Get n in NCHWnc. NCHWnc format is used in conv2dfusion expander in lite. n is the value of tiling channel.
   * @example
   *   input
   *     NCHW16c
   *   output
   *     16
   */
  static int64_t GetChannelInConvFormat(const std::string &format_string);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_UTILS_H_
