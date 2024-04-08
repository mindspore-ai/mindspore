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

#ifndef MINDSPORE_CORE_UTILS_COMPILE_CONFIG_H_
#define MINDSPORE_CORE_UTILS_COMPILE_CONFIG_H_

#include <string>
#include <memory>
#include <map>
#include <functional>
#include "mindapi/base/macros.h"

namespace mindspore {
class MS_CORE_API CompileConfigManager {
 public:
  using CompileConfigCollectFunc = std::function<std::map<std::string, std::string>()>;

  /// \brief Get instance of CompileConfigManager.
  ///
  /// \return Instance of CompileConfigManager.
  static CompileConfigManager &GetInstance() noexcept;

  /// \brief Disable the default constructor.
  CompileConfigManager(const CompileConfigManager &) = delete;
  /// \brief Disable the default copy constructor.
  CompileConfigManager &operator=(const CompileConfigManager &) = delete;
  /// \brief Destructor.
  ~CompileConfigManager() = default;

  static void set_collect_func(CompileConfigCollectFunc func) { collect_func_ = func; }

  void CollectCompileConfig();

  void SetConfig(const std::string &config_name, const std::string &value, bool overwrite = true);

  std::string GetConfig(const std::string &config_name);

 private:
  CompileConfigManager() = default;
  inline static CompileConfigCollectFunc collect_func_{nullptr};
  std::map<std::string, std::string> compile_config_;
  bool collect_finished_{false};
};

namespace common {
MS_CORE_API std::string GetCompileConfig(const std::string &config_name);
MS_CORE_API void SetCompileConfig(const std::string &config_name, const std::string &value, bool overwrite = true);
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_COMPILE_CONFIG_H_
