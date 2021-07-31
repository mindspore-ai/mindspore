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
#include <utility>
#include <memory>
#include "include/lite_utils.h"

namespace mindspore {
namespace opt {
/// \brief PassPosition defined where to plae user's pass.
enum MS_API PassPosition { POSITION_BEGIN = 0, POSITION_END = 1 };

/// \brief P defined a basic interface.
///
/// \note List public class and interface for reference.
class MS_API Pass;
using PassPtr = std::shared_ptr<Pass>;

/// \brief PassRegistry defined registration of Pass.
class MS_API PassRegistry {
 public:
  /// \brief Constructor of PassRegistry to register pass.
  ///
  /// \param[in] pos Define where to replace the pass.
  /// \param[in] pass Define user's defined pass.
  PassRegistry(const std::string &pass_name, const PassPtr &pass);

  /// \brief Constructor of PassRegistry to assign which passes are required for external extension.
  ///
  /// \param[in position Define the place where assigned passes will run.
  /// \param[in] assigned Define the name of passes assigned by user.
  PassRegistry(PassPosition position, const std::vector<std::string> &assigned);

  /// \brief Destructor of PassRegistrar.
  ~PassRegistry() = default;
};

/// \brief Defined registering macro to register Pass, which called by user directly.
///
/// \param[in] name Define name of user's pass, which is a string.
/// \param[in] pass Define user's defined pass.
#define REG_PASS(name, pass) static PassRegistry g_##name##PassReg(#name, std::make_shared<pass>());

/// \brief Defined assigning macro to assign Passes, which called by user directly.
///
/// \param[in] position Define the place where assigned passes will run.
/// \param[in] assigned Define the name of passes assigned by user.
#define REG_SCHEDULED_PASS(position, assigned) static PassRegistry g_##position(position, assigned);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_REGISTRY_H_
