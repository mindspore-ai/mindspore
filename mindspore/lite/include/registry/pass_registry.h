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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_REGISTRY_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_REGISTRY_H_

#include <vector>
#include <string>
#include <memory>
#include "include/api/types.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
namespace registry {
class PassBase;
using PassBasePtr = std::shared_ptr<PassBase>;
/// \brief PassPosition defined where to place user's pass.
enum MS_API PassPosition { POSITION_BEGIN = 0, POSITION_END = 1 };

/// \brief PassRegistry defined registration of Pass.
class MS_API PassRegistry {
 public:
  /// \brief Constructor of PassRegistry to register pass.
  ///
  /// \param[in] pass_name Define the name of the pass, a string which should guarantee uniqueness.
  /// \param[in] pass Define pass instance.
  inline PassRegistry(const std::string &pass_name, const PassBasePtr &pass);

  /// \brief Constructor of PassRegistry to assign which passes are required for external extension.
  ///
  /// \param[in] position Define the place where assigned passes will run.
  /// \param[in] names Define the names of the passes.
  inline PassRegistry(PassPosition position, const std::vector<std::string> &names);

  /// \brief Destructor of PassRegistrar.
  ~PassRegistry() = default;

  /// \brief Static method to obtain external scheduling task assigned by user.
  ///
  /// \param[in] position Define the place where assigned passes will run.
  ///
  /// \return Passes' Name Vector.
  inline static std::vector<std::string> GetOuterScheduleTask(PassPosition position);

  /// \brief Static method to obtain pass instance according to passes' name.
  ///
  /// \param[in] pass_name Define the name of pass.
  ///
  /// \return Pass Instance Vector.
  inline static PassBasePtr GetPassFromStoreRoom(const std::string &pass_name);

 private:
  PassRegistry(const std::vector<char> &pass_name, const PassBasePtr &pass);
  PassRegistry(PassPosition position, const std::vector<std::vector<char>> &names);
  static std::vector<std::vector<char>> GetOuterScheduleTaskInner(PassPosition position);
  static PassBasePtr GetPassFromStoreRoom(const std::vector<char> &pass_name_char);
};

PassRegistry::PassRegistry(const std::string &pass_name, const PassBasePtr &pass)
    : PassRegistry(StringToChar(pass_name), pass) {}

PassRegistry::PassRegistry(PassPosition position, const std::vector<std::string> &names)
    : PassRegistry(position, VectorStringToChar(names)) {}

std::vector<std::string> PassRegistry::GetOuterScheduleTask(PassPosition position) {
  return VectorCharToString(GetOuterScheduleTaskInner(position));
}

PassBasePtr PassRegistry::GetPassFromStoreRoom(const std::string &pass_name) {
  return GetPassFromStoreRoom(StringToChar(pass_name));
}
/// \brief Defined registering macro to register Pass, which called by user directly.
///
/// \param[in] name Define the name of the pass, a string which should guarantee uniqueness.
/// \param[in] pass Define pass instance.
#define REG_PASS(name, pass) \
  static mindspore::registry::PassRegistry g_##name##PassReg(#name, std::make_shared<pass>());

/// \brief Defined assigning macro to assign Passes, which called by user directly.
///
/// \param[in] position Define the place where assigned passes will run.
/// \param[in] names Define the names of the passes.
#define REG_SCHEDULED_PASS(position, names) static mindspore::registry::PassRegistry g_##position(position, names);
}  // namespace registry
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_REGISTRY_H_
