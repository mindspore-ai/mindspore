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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDIR_CONTROL_FLOW_ADJUST_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDIR_CONTROL_FLOW_ADJUST_H_

#include <string>
#include <vector>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_params.h"
#include "include/registry/converter_context.h"

using mindspore::converter::FmkType;
using mindspore::lite::quant::QuantType;
namespace mindspore::lite {
class MindIRControlFlowAdjust {
 public:
  MindIRControlFlowAdjust() {}
  ~MindIRControlFlowAdjust() = default;
  void SetFmkType(FmkType fmk_type) { fmk_type_ = fmk_type; }
  bool Run(const FuncGraphPtr &graph);

 private:
  std::vector<AnfNodePtr> GetFgOutput(const FuncGraphPtr &fg);
  int ModifyFgToCallAfterFg(const FuncGraphPtr &fg, const FuncGraphPtr &after_fg);
  bool HasCallAfter(const FuncGraphPtr &partial_fg);
  FuncGraphPtr AddAfterFuncGraph(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &one_of_inline_fg_output);
  int InsertPartialFusionForRawCall(const std::set<FuncGraphPtr> &all_func_graphs);
  int ResetFuncGraph(const FuncGraphPtr &fg, std::set<FuncGraphPtr> all_func_graphs);
  int MoveCallInputsToPartialFusionInputs(const std::set<FuncGraphPtr> &all_func_graphs);

  FmkType fmk_type_ = FmkType::kFmkTypeMs;
  int status_ = RET_OK;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDIR_CONTROL_FLOW_ADJUST_H_
