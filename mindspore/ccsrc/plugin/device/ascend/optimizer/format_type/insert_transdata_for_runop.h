/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_INSERT_TRANSDATA_FOR_RUNOP_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_INSERT_TRANSDATA_FOR_RUNOP_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class RunOpInsertTransData : public Pass {
 public:
  RunOpInsertTransData() : Pass("insert_transdata_for_runop"), kernel_select_(std::make_shared<KernelSelect>()) {}
  ~RunOpInsertTransData() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool InsertTransdataForOutput(const FuncGraphPtr &graph);
  bool ConvertNodeFormat(const FuncGraphPtr &graph, const AnfNodePtr &node, const std::string &format,
                         size_t insert_index, size_t input_index, bool is_insert) const;
  KernelSelectPtr kernel_select_;
  ShapeVector input_shape_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_INSERT_TRANSDATA_FOR_RUNOP_H_
