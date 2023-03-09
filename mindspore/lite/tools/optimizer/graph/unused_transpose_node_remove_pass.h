/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_TRANSPOSE_NODE_REMOVE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_TRANSPOSE_NODE_REMOVE_PASS_H_
#include <string>
#include "include/backend/optimizer/pass.h"
#include "include/registry/converter_context.h"

using mindspore::converter::FmkType;
namespace mindspore::opt {
class RemoveUnusedTransposeOpPass : public Pass {
 public:
  RemoveUnusedTransposeOpPass() : Pass("remove_unused_cast_pass") {}
  ~RemoveUnusedTransposeOpPass() override = default;
  void SetFmkType(FmkType fmkType);
  bool Run(const FuncGraphPtr &graph) override;

 private:
  FmkType fmk_type = converter::kFmkTypeTf;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_TRANSPOSE_NODE_REMOVE_PASS_H_
