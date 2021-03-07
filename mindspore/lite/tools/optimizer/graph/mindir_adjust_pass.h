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

#include <string>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/param_value_lite.h"

using mindspore::lite::converter::FmkType;
using mindspore::schema::QuantType;
namespace mindspore::opt {
class MindirAdjustPass : public Pass {
 public:
  MindirAdjustPass() : Pass("mindir_adjust_pass") {}
  ~MindirAdjustPass() override = default;
  void SetQuantType(QuantType quant_type) { quant_type_ = quant_type; }
  void SetFmkType(FmkType fmk_type) { fmk_type_ = fmk_type; }
  int ValueNodeInt64Convert(AnfNodePtr anf_node);
  void SetTrainFlag(bool train_flag) { train_flag_ = train_flag; }
  int ParameterNodeConvert(AnfNodePtr anf_node);
  int ComputeQuantParams(AnfNodePtr anf_node);
  bool Run(const FuncGraphPtr &graph) override;

 protected:
  QuantType quant_type_ = QuantType::QuantType_QUANT_NONE;
  FmkType fmk_type_ = FmkType::FmkType_MS;
  bool train_flag_ = false;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MINDIR_ADJUST_PASS_H_
