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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PREPROCESS_DYNAMIC_SHAPE_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PREPROCESS_DYNAMIC_SHAPE_H

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace opt {
class DynamicShapePreprocessor {
  typedef std::map<AnfNodePtr, std::pair<std::vector<ShapeVector>, std::vector<ShapeVector>>> ShapeContainer;

 public:
  DynamicShapePreprocessor() = default;
  ~DynamicShapePreprocessor() = default;
  int Run(const FuncGraphPtr &func_graph);
  const ShapeContainer &GetShapeContainer() { return op_shape_infos_; }

 private:
  bool CheckIsDynamicModel(const FuncGraphPtr &func_graph);
  int ProcessOps(const FuncGraphPtr &func_graph);
  int DoInfer(const CNodePtr &cnode, const std::string &op_type);
  ShapeContainer op_shape_infos_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PREPROCESS_DYNAMIC_SHAPE_H
