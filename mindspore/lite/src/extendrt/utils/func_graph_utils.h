/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_FUNC_GRAPH_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_FUNC_GRAPH_UTILS_H_

#include <utility>
#include <string>
#include <vector>
#include <tuple>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/func_graph.h"
#include "include/api/data_type.h"
#include "include/api/status.h"
#include "mindspore/ccsrc/kernel/kernel.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
using AnfWithOutIndex = std::pair<AnfNodePtr, size_t>;
using kernel::BaseOperatorPtr;

class FuncGraphUtils {
 public:
  static tensor::TensorPtr GetConstNodeValue(AnfNodePtr input_node);
  static std::vector<common::KernelWithIndex> GetNodeInputs(const AnfNodePtr &anf_node);

  static bool GetCNodeOperator(const CNodePtr &cnode, BaseOperatorPtr *base_operator);

  static bool GetCNodeInputsOutputs(const CNodePtr &cnode, std::vector<AnfWithOutIndex> *input_tensors,
                                    std::vector<AnfWithOutIndex> *output_tensors);
  static bool GetFuncGraphInputs(const FuncGraphPtr &func_graph, std::vector<AnfWithOutIndex> *inputs);
  static bool GetFuncGraphOutputs(const FuncGraphPtr &func_graph, std::vector<AnfWithOutIndex> *outputs);

  static DataType GetTensorDataType(const AnfWithOutIndex &tensor);
  static ShapeVector GetTensorShape(const AnfWithOutIndex &tensor);
  static std::string GetTensorName(const AnfWithOutIndex &tensor);
  static AbstractBasePtr GetAbstract(const AnfWithOutIndex &tensor);

  static void GetFuncGraphInputsInfo(const FuncGraphPtr &graph, std::vector<tensor::TensorPtr> *inputs,
                                     std::vector<std::string> *inputs_name);
  static void GetFuncGraphOutputsInfo(const FuncGraphPtr &graph, std::vector<tensor::TensorPtr> *outputs,
                                      std::vector<std::string> *output_names);
  static Status UnifyGraphToNHWCFormat(const FuncGraphPtr &graph);

  static std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> TransformSegmentToAnfGraph(const AnfNodePtrList &lst);

  static AnfNodePtrList GetOutput(const AnfNodePtrList &nodes, const NodeUsersMap &users,
                                  const mindspore::HashSet<AnfNodePtr> &seen);
  static AnfNodePtr RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node, AnfNodePtrList *inputs_ptr,
                                    mindspore::HashMap<AnfNodePtr, AnfNodePtr> *eqv_ptr);

 private:
  static ValuePtr GetNodeValuePtr(AnfNodePtr input_node);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_FUNC_GRAPH_UTILS_H_
