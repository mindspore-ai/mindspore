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
#ifndef MINDSPORE_CORE_OPS_ENCODER_LAYER_H_
#define MINDSPORE_CORE_OPS_ENCODER_LAYER_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "kernel/cpu/nnacl/op_base.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEncoderLayer = "EncoderLayer";
/// \brief EncoderLayer op in MindIR.
class OPS_API EncoderLayer : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EncoderLayer);
  /// \brief Constructor.
  EncoderLayer() : BaseOperator(kNameEncoderLayer) {
    InitIOName({"input", "gamma1", "beta1", "weight_attn_qkv", "bias_attn_qkv", "mask", "weight_attn_o", "bias_attn_o",
                "gamma2", "beta2", "weight_m", "bias_m", "weight_p", "bias_p"},
               {"output"});
  }
  /// \brief Initialize EncoderLayer op.
  /// \param[in] head_num Define head number.
  /// \param[in] head_size Define size per head.
  /// \param[in] eps_layernorm1 Define eps layernorm1.
  /// \param[in] eps_layernorm2 Define eps layernorm2.
  /// \param[in] eps_layernorm3 Define eps layernorm3.
  /// \param[in] ffn_hidden_size Define ffn hidden size.
  /// \param[in] expert_num Define expert num.
  /// \param[in] expert_offset_id Define expert_offset_id.
  /// \param[in] capacity_factor Define capacity_factor.
  /// \param[in] position_bias Define position_bias.
  /// \param[in] scale Define scale.
  /// \param[in] act_type Define act_type.
  /// \param[in] layer_norm Define act_type.
  /// \param[in] use_past Define use_past.
  /// \param[in] query_layer Define query_layer.
  /// \param[in] moe Define moe.
  /// \param[in] embedding_layer Define embedding_layer.

  void Init(int64_t head_num, int64_t head_size, float eps_layernorm1, float eps_layernorm2, float eps_layernorm3,
            int64_t ffn_hidden_size, int64_t expert_num, int64_t expert_offset_id, float capacity_factor,
            bool position_bias, bool post_layernorm, float scale = 1.0f, ActType act_type = ActType::ActType_Gelu,
            bool layer_norm = false, bool use_past = false, bool query_layer = false, bool moe = false,
            bool embedding_layer = false);
  void set_head_num(int64_t head_num);
  void set_head_size(int64_t head_size);
  void set_post_layernorm(bool post_layernorm);
  void set_eps_layernorm1(float eps_layernorm1);
  void set_eps_layernorm2(float eps_layernorm2);
  void set_eps_layernorm3(float eps_layernorm3);
  void set_ffn_hidden_size(int64_t ffn_hidden_size);
  void set_expert_num(int64_t expert_num);
  void set_expert_offset_id(int64_t expert_offset_id);
  void set_capacity_factor(float capacity_factor);
  void set_position_bias(bool position_bias);
  void set_scale(float scale);
  void set_act_type(ActType act_type);
  void set_layer_norm(bool layer_norm);
  void set_use_past(bool use_past);
  void set_query_layer(bool query_layer);
  void set_moe(bool moe);
  void set_embedding_layer(bool embedding_layer);

  int64_t get_head_num() const;
  int64_t get_head_size() const;
  bool get_post_layernorm() const;
  float get_eps_layernorm1() const;
  float get_eps_layernorm2() const;
  float get_eps_layernorm3() const;
  int64_t get_ffn_hidden_size() const;
  int64_t get_expert_num() const;
  int64_t get_expert_offset_id() const;
  float get_capacity_factor() const;
  bool get_position_bias() const;
  float get_scale() const;
  ActType get_act_type() const;
  bool get_layer_norm() const;
  bool get_use_past() const;
  bool get_query_layer() const;
  bool get_moe() const;
  bool get_embedding_layer() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_ENCODER_LAYER_H_
