/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_UTILS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_UTILS_H_

#include <memory>
#include "ir/anf.h"
#include "include/api/context.h"
#include "include/registry/converter_context.h"
#include "schema/inner/model_generated.h"
#include "src/litert/inner_context.h"

namespace mindspore {
namespace opt {
class ConstFoldProcessor {
 public:
  explicit ConstFoldProcessor(converter::FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : fmk_type_(fmk_type), train_flag_(train_flag) {}
  int DoConstantFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  ~ConstFoldProcessor() = default;

 private:
  bool Init();
  converter::FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  std::shared_ptr<lite::InnerContext> context_{nullptr};
  std::shared_ptr<mindspore::Context> ms_context_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_CONST_FOLD_FOLD_UTILS_H_
