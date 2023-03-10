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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FORMAT_TYPE_ELIMINATE_GRAPH_OUTPUT_TRANSDATA_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FORMAT_TYPE_ELIMINATE_GRAPH_OUTPUT_TRANSDATA_H_

#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class EliminateGraphOutputTransdata : public Pass {
 public:
  EliminateGraphOutputTransdata() : Pass("eliminate_graph_output_transdata") {}
  ~EliminateGraphOutputTransdata() = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FORMAT_TYPE_ELIMINATE_GRAPH_OUTPUT_TRANSDATA_H_
