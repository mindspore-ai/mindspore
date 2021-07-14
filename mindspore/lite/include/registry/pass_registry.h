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
#include <mutex>
#include <memory>
#include <unordered_map>
#include "include/lite_utils.h"

namespace mindspore {
namespace opt {
/// \brief PassPosition defined where to place user's pass.
enum MS_API PassPosition { POSITION_BEGIN = 0, POSITION_END = 1 };

/// \brief P defined a basic interface.
///
/// \note List public class and interface for reference.
class MS_API Pass;
using PassPtr = std::shared_ptr<Pass>;

/// \brief PassRegistry defined registration of Pass.
class MS_API PassRegistry {
 public:
  /// \brief Destructor of PassRegistry.
  virtual ~PassRegistry() = default;

  /// \brief Static method to get a single instance of PassRegistry.
  ///
  /// \return Pointer of PassRegistry.
  static PassRegistry *GetInstance();

  /// \brief Method to register Pass.
  ///
  /// \param[in] position Define where to replace the pass.
  /// \param[in] pass Define user's defined pass.
  void RegPass(int position, const PassPtr &pass);

  /// \brief Method to get all passes user write.
  ///
  /// \return A map include all pass.
  const std::unordered_map<int, PassPtr> &GetPasses() const;

 private:
  /// \brief Constructor of PassRegistry.
  PassRegistry() = default;

 private:
  std::unordered_map<int, PassPtr> passes_;
  std::mutex mutex_;
};

/// \brief PassRegistrar defined registration class of Pass.
class MS_API PassRegistrar {
 public:
  /// \brief Constructor of PassRegistrar to register pass.
  ///
  /// \param[in] pos Define where to replace the pass.
  /// \param[in] pass Define user's defined pass.
  PassRegistrar(int pos, const PassPtr &pass) { PassRegistry::GetInstance()->RegPass(pos, pass); }

  /// \brief Destructor of PassRegistrar.
  ~PassRegistrar() = default;
};

/// \brief Defined registering macro to register Pass, which called by user directly.
///
/// \param[in] position Define where to replace the pass.
/// \param[in] pass Define user's defined pass.
#define REG_PASS(position, pass) static PassRegistrar g_##position##PassReg(position, std::make_shared<pass>());

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_REGISTRY_H_
