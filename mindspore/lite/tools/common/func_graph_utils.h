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

#ifndef MINDSPORE_LITE_TOOL_COMMON_FUNC_GRAPH_UTILS_H_
#define MINDSPORE_LITE_TOOL_COMMON_FUNC_GRAPH_UTILS_H_

#include <string>
#include <vector>
#include <utility>
#include "ir/func_graph.h"

namespace mindspore {
class FuncGraphUtils {
 public:
  static std::vector<std::string> GetFuncGraphOutputNames(const FuncGraphPtr &func_graph);
  static void SetFuncGraphOutputNames(const FuncGraphPtr &func_graph, const std::vector<std::string> &output_names);
  static AbstractBasePtr GetAbstractFromNode(const std::pair<AnfNodePtr, int64_t> &node);
  static void SetOutputName(const std::pair<AnfNodePtr, int64_t> &node, const std::string &name);
  static std::string GetOutputName(const std::pair<AnfNodePtr, int64_t> &node_index);
  static tensor::TensorPtr GetParameterConstValue(const AnfNodePtr &anf_node);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOL_COMMON_FUNC_GRAPH_UTILS_H_
