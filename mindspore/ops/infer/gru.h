/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRU_H_
#define MINDSPORE_CORE_OPS_GRU_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGRU = "GRU";
/// \brief GRU defined the GRU operator prototype.
class OPS_API GRU : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GRU);
  /// \brief Constructor.
  GRU() : BaseOperator(kNameGRU) {
    InitIOName({"x", "weight_input", "weight_hidden", "bias_input", "bias_hidden", "seq_length", "init_h"},
               {"output", "output_h", "update", "reset", "new", "hidden_new"});
  }
  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] bidirectional Define a boolean value to indicate whether the gru is single or double direction.
  void Init(bool bidirectional = false);

  /// \brief Method to set bidirectional attribute.
  ///
  /// \param bidirectional Define a boolean value to indicate whether the gru is single or double direction.
  void set_bidirectional(bool bidirectional);

  /// \brief Method to get bidirectional attribute.
  ///
  /// \return a boolean value.
  bool get_bidirectional() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRU_H_
