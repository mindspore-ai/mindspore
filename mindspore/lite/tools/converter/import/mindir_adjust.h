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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_ADJUST_PASS_H_

#include <set>
#include <string>
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"

using mindspore::converter::FmkType;
using mindspore::schema::QuantType;
namespace mindspore::lite {
class MindirAdjust {
 public:
  MindirAdjust() {}
  ~MindirAdjust() = default;
  void SetFmkType(FmkType fmk_type) { fmk_type_ = fmk_type; }
  void SetTrainFlag(bool train_flag) { train_flag_ = train_flag; }
  bool Run(const FuncGraphPtr &graph);

 private:
  int ValueNodeInt64Convert(AnfNodePtr anf_node);
  int ComputeQuantParams(AnfNodePtr anf_node);
  int UpdateConv2DTransposeInput(const CNodePtr &cnode);
  int ResetFuncGraph(const FuncGraphPtr &fg, std::set<FuncGraphPtr> all_func_graphs);

  FmkType fmk_type_ = FmkType::kFmkTypeMs;
  bool train_flag_ = false;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_ADJUST_PASS_H_
