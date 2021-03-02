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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SHAPE_OPS_SPLITTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SHAPE_OPS_SPLITTER_H_
#include <memory>
#include <vector>
#include <utility>
#include "ir/func_graph.h"
#include "backend/optimizer/common/pass.h"

namespace mindspore {
namespace opt {
class ShapeOpsSplitter : public Pass {
 public:
  explicit ShapeOpsSplitter(std::vector<PrimitivePtr> shape_ops)
      : Pass("shape_ops_splitter"), shape_ops_(std::move(shape_ops)) {}
  ~ShapeOpsSplitter() override = default;
  bool Run(const FuncGraphPtr &func_graph);

 private:
  bool Process(const FuncGraphPtr &func_graph);
  bool IsMultiUserShapeOps(const AnfNodePtr &node, const FuncGraphManagerPtr &mng);
  std::vector<PrimitivePtr> shape_ops_;
};
using ShapeOpsSplitterPtr = std::shared_ptr<ShapeOpsSplitter>;
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SHAPE_OPS_SPLITTER_H_
