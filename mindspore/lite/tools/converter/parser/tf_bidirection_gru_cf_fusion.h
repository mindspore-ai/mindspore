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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_BIDIRECTION_GRU_CF_FUSION_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_BIDIRECTION_GRU_CF_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include "schema/inner/model_generated.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
// fuse tf 1.x bidirection_gru into MSLITE GRU
class TfBidirectionGruCfFusion : public TfBidirectionGruFusion {
 public:
  explicit TfBidirectionGruCfFusion(const std::string &name = "tf_bidirection_gru_cf_fusion", bool multi_graph = true);

  ~TfBidirectionGruCfFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  BaseRef DefineGruCellPattern(const BaseRef &in_ta_read, const BaseRef &switch3_true,
                               const std::vector<VarPtr> &vars) const;

  const BaseRef DefineBidirectionRnnPattern(const BaseRef &input, const std::vector<VarPtr> &vars,
                                            const VarPtr &init_state) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_BIDIRECTION_GRU_CF_FUSION_H_
