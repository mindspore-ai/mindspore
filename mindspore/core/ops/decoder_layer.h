/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_DECODER_LAYER_H_
#define MINDSPORE_CORE_OPS_DECODER_LAYER_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDecoderLayer = "DecoderLayer";
/// \brief DecoderLayer op in MindIR.
class MIND_API DecoderLayer : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DecoderLayer);
  /// \brief Constructor.
  DecoderLayer() : BaseOperator(kNameDecoderLayer) {
    InitIOName({"input",
                "gamma1",
                "beta1",
                "weight_qkv",
                "bias_attn_qkv",
                "input_mask",
                "weight_attn_o",
                "bias_attn_o",
                "gamma2",
                "beta2",
                "encoder_output",
                "weight_attn_q",
                "weight_attn_kv",
                "bias_attn_cross_qkv",
                "cross_mask",
                "weight_attn_cross_o",
                "bias_attn_cross_o",
                "gamma3",
                "beta3",
                "weight_m",
                "bias_m",
                "weight_p",
                "bias_p"},
               {"output"});
  }
  /// \brief Initialize DecoderLayer op.
  /// \param[in] head_num Define head number.
  /// \param[in] head_size Define size per head.
  /// \param[in] eps_layernorm1 Define eps layernorm1.
  /// \param[in] eps_layernorm2 Define eps layernorm2.
  /// \param[in] eps_layernorm3 Define eps layernorm3.
  /// \param[in] eps_layernorm4 Define eps layernorm4.
  /// \param[in] ffn_hidden_size Define ffn hidden size.
  /// \param[in] position_bias1 Define position_bias1.
  /// \param[in] position_bias2 Define position_bias2.
  /// \param[in] scale1 Define scale1.
  /// \param[in] scale2 Define scale2.
  /// \param[in] act_type Define act_type.
  /// \param[in] layer_norm Define act_type.
  void Init(int64_t head_num, int64_t head_size, float eps_layernorm1, float eps_layernorm2, float eps_layernorm3,
            float eps_layernorm4, int64_t ffn_hidden_size, bool position_bias1, bool position_bias2,
            bool post_layernorm, float scale1 = 1.0f, float scale2 = 1.0f, ActType act_type = ActType::ActType_Gelu,
            bool layer_norm = false);
  void set_head_num(int64_t head_num);
  void set_head_size(int64_t head_size);
  void set_post_layernorm(bool post_layernorm);
  void set_eps_layernorm1(float eps_layernorm1);
  void set_eps_layernorm2(float eps_layernorm2);
  void set_eps_layernorm3(float eps_layernorm3);
  void set_eps_layernorm4(float eps_layernorm4);
  void set_ffn_hidden_size(int64_t ffn_hidden_size);
  void set_position_bias1(bool position_bias1);
  void set_position_bias2(bool position_bias2);
  void set_scale1(float scale1);
  void set_scale2(float scale2);
  void set_act_type(ActType act_type);
  void set_layer_norm(bool layer_norm);
  int64_t get_head_num() const;
  int64_t get_head_size() const;
  bool get_post_layernorm() const;
  float get_eps_layernorm1() const;
  float get_eps_layernorm2() const;
  float get_eps_layernorm3() const;
  float get_eps_layernorm4() const;
  int64_t get_ffn_hidden_size() const;
  bool get_position_bias1() const;
  bool get_position_bias2() const;
  float get_scale1() const;
  float get_scale2() const;
  ActType get_act_type() const;
  bool get_layer_norm() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DECODER_LAYER_H_
