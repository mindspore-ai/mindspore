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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_LSTM_CELL_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_LSTM_CELL_FUSION_H_
#include <vector>
#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
class TfLstmCellFusion : public TfliteLstmCellFusion {
 public:
  explicit TfLstmCellFusion(const std::string &name = "TfLstmCellFusion", bool multigraph = true);
  ~TfLstmCellFusion() override = default;

 private:
  AnfNodePtr GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const override;
  CNodePtr CreateLSTMNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const EquivPtr &body_equiv,
                          const std::string &base_name, float zoneout_cell, float zoneout_hidden) const override;

  static lite::STATUS SplitWeights(const AnfNodePtr &weight, const ParameterPtr &weight_i, const ParameterPtr &weight_c,
                                   int hidden_size);
  static lite::STATUS SetWeightAbstractAndDefault(const ParameterPtr &weight, const std::vector<int64_t> &shape,
                                                  const float *data_ptr, int hidden_size);
  lite::STATUS PopulateBiasNode(const EquivPtr &body_equiv, const ParameterPtr &new_bias, const AnfNodePtr &old_bias,
                                int hidden_size) const;

 private:
  mutable VarPtr forget_bias_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_LSTM_CELL_FUSION_H_
