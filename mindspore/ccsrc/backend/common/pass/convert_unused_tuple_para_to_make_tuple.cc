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

#include "backend/common/pass/convert_unused_tuple_para_to_make_tuple.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool ConvertUnusedTupleParaToMakeTuple::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = dyn_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &input : kernel_graph->inputs()) {
    auto parameter = dyn_cast<Parameter>(input);
    // Only deal with the unused parameter with tuple outputs.
    if (parameter == nullptr || !common::AnfAlgo::IsTupleOutput(parameter)) {
      continue;
    }
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    if (manager->node_users().find(parameter) != manager->node_users().end()) {
      continue;
    }
    if (kernel_graph->FindTupleParameterToMakeTupleMap(parameter) != nullptr) {
      continue;
    }
    auto make_tuple = kernel_graph->TransTupleToMakeTuple(parameter);
    kernel_graph->InsertTupleParameterToMakeTupleMap(parameter, make_tuple);
    // Replace graph inputs.
    kernel_graph->ReplaceGraphInput(parameter, make_tuple);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
