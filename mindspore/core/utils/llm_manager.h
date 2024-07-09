/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_LLM_MANAGER_H_
#define MINDSPORE_CORE_UTILS_LLM_MANAGER_H_

#include <string>
#include <memory>
#include <map>
#include "mindapi/base/macros.h"

namespace mindspore {
// Current not support multi -thread use this Single Instance
class MS_CORE_API LLMManager {
 public:
  /// \brief Get instance of LLMManager.
  ///
  /// \return Instance of LLMManager.
  static LLMManager &GetInstance() noexcept;

  /// \brief Disable the default copy constructor.
  LLMManager &operator=(const LLMManager &) = delete;
  /// \brief Destructor.
  ~LLMManager() = default;

  /// \brief Update the current round_up_max_seq_length.
  ///
  /// \param[in] The max_seq_length of an obj to be compiled.
  /// \return The result of update, if no change, return false, if change, return true
  bool update_round_up_max_seq_length(int32_t max_seq_lenght);

  /// \brief Get the current round_up_max_seq_length.
  ///
  /// \return The current round_up_max_seq_length.
  int32_t get_current_round_up_max_seq_length();

  /// \brief Get the seq_length_graph_input_index.
  ///
  /// \return The seq_length_graph_input_index.
  int32_t get_seq_length_graph_input_index();

  bool enable_multi_level_seq_length_{false};

 private:
  LLMManager();

  void init();

 private:
  bool inited_{false};
  int32_t current_round_up_max_seq_length_{1024};
  int32_t seq_length_level_size_{128};
  int32_t seq_length_graph_input_index_{-1};
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_LLM_MANAGER_H_
