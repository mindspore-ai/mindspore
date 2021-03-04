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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_BIDIRECTION_GRU_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_BIDIRECTION_GRU_FUSION_H_
#include <vector>
#include <memory>
#include <string>
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "schema/inner/model_generated.h"
#include "src/param_value_lite.h"
#include "backend/optimizer/common/optimizer.h"
#include "utils/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
constexpr size_t kWhileUniqInputsLength = 6;
// fuse tf 2.x bidirection_gru into MSLITE GRU
class TfBidirectionGruFusion : public PatternProcessPass {
 public:
  explicit TfBidirectionGruFusion(int num_fw_vars = kWhileUniqInputsLength, int num_bw_vars = kWhileUniqInputsLength,
                                  const std::string &name = "tf_bidirection_gru_fusion", bool multi_graph = true);
  ~TfBidirectionGruFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  virtual AnfNodePtr GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const;
  CNodePtr CreateBiDirectionGruNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const EquivPtr &equiv,
                                    const std::string &base_name, int var_offset) const;
  CNodePtr GetPostProcessNode(const FuncGraphPtr &func_graph, const CNodePtr &gru_output,
                              const std::string base_name) const;

 private:
  AnfNodePtr GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const;

  ParamValueLitePtr GetDefaultParamValue(const AnfNodePtr &parameter_anf) const;
  lite::STATUS GetInputAndHiddenSize(const AnfNodePtr &fw_cand_kernel_anf, const AnfNodePtr &bw_cand_kernel_anf,
                                     int *input_size, int *hidden_size) const;
  ParameterPtr AddDefaultParameter(const FuncGraphPtr &func_graph, const std::string &name,
                                   const std::vector<int> &shape, const TypeId type, void **tensor_data) const;
  lite::STATUS ConvertWeightData(const AnfNodePtr &gate_weight, const AnfNodePtr &cand_weight, const int input_size,
                                 const int hidden_size, float *gate_tensor_data, float *recu_tensor_data) const;
  lite::STATUS ConvertBiasData(const AnfNodePtr &gate_bias, const AnfNodePtr &cand_bias, const int hidden_size,
                               float *tensor_data) const;
  void CopyFlattenMatData(const float *mat, const int R, const int C, const int r0, const int r1, const int c0,
                          const int c1, float *data, bool t = false) const;
  CNodePtr GetStackedHiddenState(const FuncGraphPtr &func_graph, const AnfNodePtr &fw_init_state,
                                 const AnfNodePtr &bw_init_state, const std::string base_name) const;

 protected:
  std::vector<VarPtr> fw_vars_;
  std::vector<VarPtr> bw_vars_;
  VarPtr input_;
  VarPtr input_length_;
  VarPtr transpose_input_;
  VarPtr fw_init_state_;
  VarPtr bw_init_state_;
};
inline bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }

inline bool IsOpType(const BaseRef &n, const PrimitivePtr &prim) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_BIDIRECTION_GRU_FUSION_H_
