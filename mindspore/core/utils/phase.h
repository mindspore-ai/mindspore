/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_PHASE_H_
#define MINDSPORE_CORE_UTILS_PHASE_H_

#include <string>
#include <memory>
#include <map>
#include "mindapi/base/macros.h"

namespace mindspore {
class MS_CORE_API PhaseManager {
 public:
  /// \brief Get instance of PhaseManager.
  ///
  /// \return Instance of PhaseManager.
  static PhaseManager &GetInstance() noexcept;

  /// \brief Disable the default constructor.
  PhaseManager(const PhaseManager &) = delete;
  /// \brief Disable the default copy constructor.
  PhaseManager &operator=(const PhaseManager &) = delete;
  /// \brief Destructor.
  ~PhaseManager() = default;

  /// \brief Set the phase.
  ///
  /// \param[in] The phase of an obj to be compiled.
  void set_phase(const std::string &phase) { phase_ = phase; }

  /// \brief Get the current phase.
  ///
  /// \return The current phase.
  const std::string &phase() const { return phase_; }

  /// \brief Clear the phase by set an empty string.
  void ClearPhase() { phase_ = ""; }

  /// \brief Set jit config.
  ///
  /// \param[in] The current jit config.
  void set_jit_config(const std::map<std::string, std::string> &jit_config) { jit_config_ = jit_config; }

  /// \brief Get the current jit config.
  ///
  /// \return The current jit config.
  const std::map<std::string, std::string> &jit_config() const { return jit_config_; }

 private:
  PhaseManager() = default;
  std::string phase_ = "";
  std::map<std::string, std::string> jit_config_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_PHASE_H_
