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
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
constexpr size_t kWhileUniqInputsLength = 6;
// fuse tf 2.x bidirection_gru into MSLITE GRU
class TfBidirectionGruFusion : public LitePatternProcessPass {
 public:
  explicit TfBidirectionGruFusion(int num_fw_vars = kWhileUniqInputsLength, int num_bw_vars = kWhileUniqInputsLength,
                                  const std::string &name = "TfBidirectionGruFusion", bool multi_graph = true)
      : LitePatternProcessPass(name, multi_graph), num_fw_vars_(num_fw_vars), num_bw_vars_(num_bw_vars) {}

  ~TfBidirectionGruFusion() override = default;

 protected:
  bool Init() const;

  const BaseRef DefinePattern() const override;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  virtual AnfNodePtr GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const;

  CNodePtr CreateBiDirectionGruNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const EquivPtr &equiv,
                                    const std::string &base_name, int var_offset) const;

  static CNodePtr GetPostProcessNode(const FuncGraphPtr &func_graph, const CNodePtr &gru_output,
                                     const std::string &base_name);

 private:
  const VectorRef DefineFowardPattern() const;

  const VectorRef DefinebackwardPattern() const;

  AnfNodePtr GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const;

  static tensor::TensorPtr GetDefaultTensorInfo(const AnfNodePtr &parameter_anf);

  static lite::STATUS GetInputAndHiddenSize(const AnfNodePtr &fw_cand_kernel_anf, const AnfNodePtr &bw_cand_kernel_anf,
                                            int *input_size, int *hidden_size);

  static ParameterPtr AddDefaultParameter(const FuncGraphPtr &func_graph, const std::string &name,
                                          const std::vector<int> &shape, TypeId type, void **tensor_data);

  static lite::STATUS ConvertWeightData(const AnfNodePtr &gate_weight, const AnfNodePtr &cand_weight, int input_size,
                                        int hidden_size, float *gate_tensor_data, float *recu_tensor_data);

  static lite::STATUS ConvertBiasData(const AnfNodePtr &gate_bias, const AnfNodePtr &cand_bias, int hidden_size,
                                      float *tensor_data);

  static void CopyFlattenMatData(const float *mat, int C, int r0, int r1, int c0, int c1, float *data, bool t = false);

  static CNodePtr GetStackedHiddenState(const FuncGraphPtr &func_graph, const AnfNodePtr &fw_init_state,
                                        const AnfNodePtr &bw_init_state, const std::string &base_name);

 protected:
  mutable std::vector<VarPtr> fw_vars_;
  mutable std::vector<VarPtr> bw_vars_;
  mutable VarPtr input_;
  mutable VarPtr input_length_;
  mutable VarPtr transpose_input_;
  mutable VarPtr fw_init_state_;
  mutable VarPtr bw_init_state_;

 private:
  int num_fw_vars_{0};
  int num_bw_vars_{0};
};
inline bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_BIDIRECTION_GRU_FUSION_H_
