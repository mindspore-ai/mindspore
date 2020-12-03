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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_ADD_ATTR_FOR_3D_GRAPH_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_ADD_ATTR_FOR_3D_GRAPH_H
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "ir/anf.h"
#include "backend/optimizer/common/pass.h"

namespace mindspore {
namespace opt {
class AddIoFormatAttrFor3DGraph : public Pass {
 public:
  explicit AddIoFormatAttrFor3DGraph(size_t groups = 1) : Pass("add_attr_for_3d_graph"), groups_(groups) {}
  ~AddIoFormatAttrFor3DGraph() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  size_t groups_ = 1;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_ADD_ATTR_FOR_3D_GRAPH_H
