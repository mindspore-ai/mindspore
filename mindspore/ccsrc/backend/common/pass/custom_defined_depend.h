/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CUSTOM_DEFINED_DEPEND_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CUSTOM_DEFINED_DEPEND_H_
#include <vector>
#include <string>
#include <memory>

#include "include/backend/optimizer/pass.h"

#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT CustomDefinedDepend : public Pass {
 public:
  CustomDefinedDepend(bool is_ge, int64_t graph_id)
      : Pass("backend_custom_depend"), is_ge_(is_ge), graph_id_(graph_id) {}
  ~CustomDefinedDepend() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool is_ge_;
  int64_t graph_id_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CUSTOM_DEFINED_DEPEND_H_
