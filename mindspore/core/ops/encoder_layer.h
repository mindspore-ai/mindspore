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
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEncoderLayer = "EncoderLayer";
/// \brief EncoderLayer op in MindIR.
class MIND_API EncoderLayer : public BaseOperator {
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
  /// \param[in] ffn_hidden_size Define ffn hidden size.
  /// \param[in] position_bias Define ffn position_bias.
  void Init(int64_t head_num, int64_t head_size, float eps_layernorm1, float eps_layernorm2, int64_t ffn_hidden_size,
            bool position_bias, bool post_layernorm);
  void set_head_num(int64_t head_num);
  void set_head_size(int64_t head_size);
  void set_post_layernorm(bool post_layernorm);
  void set_eps_layernorm1(float eps_layernorm1);
  void set_eps_layernorm2(float eps_layernorm2);
  void set_ffn_hidden_size(int64_t ffn_hidden_size);
  void set_position_bias(bool position_bias);
  int64_t get_head_num() const;
  int64_t get_head_size() const;
  bool get_post_layernorm() const;
  float get_eps_layernorm1() const;
  float get_eps_layernorm2() const;
  int64_t get_ffn_hidden_size() const;
  bool get_position_bias() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_ENCODER_LAYER_H_
