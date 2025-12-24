/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LSTM_H_
#define MINDSPORE_CORE_OPS_LSTM_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLSTM = "LSTM";
/// \brief Performs the Long Short-Term Memory (LSTM) on the input.
/// Refer to Python API @ref mindspore.ops.LSTM for more details.
class OPS_API LSTM : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LSTM);
  /// \brief Constructor.
  LSTM() : BaseOperator(kNameLSTM) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LSTM for the inputs.
  void Init(const int64_t input_size, const int64_t hidden_size, const int64_t num_layers, const bool has_bias,
            const float dropout, const bool bidirectional = false, const float zoneout_cell = 0.0f,
            const float zoneout_hidden = 0.0f);
  /// \brief Set input_size.
  void set_input_size(const int64_t input_size);
  /// \brief Get input_size.
  ///
  /// \return input_size.
  int64_t get_input_size() const;
  /// \brief Set hidden_size.
  void set_hidden_size(const int64_t hidden_size);
  /// \brief Get hidden_size.
  ///
  /// \return hidden_size.
  int64_t get_hidden_size() const;
  /// \brief Set proj_size.
  void set_proj_size(const int64_t proj_size);
  /// \brief Get proj_size.
  ///
  /// \return proj_size.
  int64_t get_proj_size() const;
  /// \brief Set num_layers.
  void set_num_layers(const int64_t num_layers);
  /// \brief Get num_layers.
  ///
  /// \return num_layers.
  int64_t get_num_layers() const;
  /// \brief Set has_bias.
  void set_has_bias(const bool has_bias);
  /// \brief Get has_bias.
  ///
  /// \return has_bias.
  bool get_has_bias() const;
  /// \brief Set dropout.
  void set_dropout(const float dropout);
  /// \brief Get dropout.
  ///
  /// \return dropout.
  float get_dropout() const;
  /// \brief Set bidirectional.
  void set_bidirectional(const bool bidirectional);
  /// \brief Get bidirectional.
  ///
  /// \return bidirectional.
  bool get_bidirectional() const;
  /// \brief Set num_directions.
  void set_num_directions(const int64_t num_directions);
  /// \brief Get num_directions.
  ///
  /// \return num_directions.
  int64_t get_num_directions() const;
  /// \brief Set zoneout_cell.
  void set_zoneout_cell(float zoneout_cell);
  /// \brief Get zoneout_cell.
  ///
  /// \return zoneout_cell.
  float get_zoneout_cell() const;
  /// \brief Set zoneout_hidden.
  void set_zoneout_hidden(float zoneout_hidden);
  /// \brief Get zoneout_hidden.
  ///
  /// \return zoneout_hidden.
  float get_zoneout_hidden() const;
  /// \brief Get good_ld.
  ///
  /// \return good_ld.
  int64_t get_good_ld(const int64_t dim, const int64_t type_size);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LSTM_H_
