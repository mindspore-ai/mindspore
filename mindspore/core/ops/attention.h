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
#ifndef MINDSPORE_CORE_OPS_ATTENTION_H_
#define MINDSPORE_CORE_OPS_ATTENTION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAttention = "Attention";
/// \brief MultiHead-Attention op in MindIR.
class MIND_API Attention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Attention);
  /// \brief Constructor.
  Attention() : BaseOperator(kNameAttention) {
    InitIOName(
      {"q", "k", "v", "weight_q", "weight_k", "weight_v", "weight_o", "bias_q", "bias_k", "bias_v", "bias_o", "mask"},
      {"output"});
  }
  /// \brief Initialize Attention op.
  /// \param[in] head_num Define head number.
  /// \param[in] head_size Define size per head.
  /// \param[in] cross Define is cross attention. Default false.
  /// \param[in] position_bias Define is position bias attention.
  void Init(int64_t head_num, int64_t head_size, bool position_bias, bool cross = false);
  void set_head_num(int64_t head_num);
  void set_head_size(int64_t head_size);
  void set_cross(bool cross);
  void set_position_bias(bool position_bias);
  int64_t get_head_num() const;
  int64_t get_head_size() const;
  bool get_cross() const;
  bool get_position_bias() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_ATTENTION_H_
