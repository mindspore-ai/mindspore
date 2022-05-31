/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PREPROCESS_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PREPROCESS_PASS_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include "include/registry/pass_base.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace dpico {
class DpicoPreprocessPass : public registry::PassBase {
 public:
  DpicoPreprocessPass() : PassBase("DpicoPreprocessPass") {}
  ~DpicoPreprocessPass() override = default;

  bool Execute(const api::FuncGraphPtr &func_graph) override;

 private:
  STATUS PreProcessBiadAdd(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode);
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PREPROCESS_PASS_H_
