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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_REGISTRY_PASS_REGISTRY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_REGISTRY_PASS_REGISTRY_H_

#include <vector>
#include <string>
#include <utility>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "include/lite_utils.h"

namespace mindspore {
namespace opt {
enum MS_API PassPosition { POSITION_BEGIN = 0, POSITION_END = 1 };

class MS_API Pass;
using PassPtr = std::shared_ptr<Pass>;
class MS_API PassRegistry {
 public:
  virtual ~PassRegistry() = default;
  static PassRegistry *GetInstance();
  void RegPass(int position, const PassPtr &pass);
  const std::unordered_map<int, PassPtr> &GetPasses() const;

 private:
  PassRegistry() = default;

 private:
  std::unordered_map<int, PassPtr> passes_;
  std::mutex mutex_;
};

class MS_API PassRegistrar {
 public:
  PassRegistrar(int pos, const PassPtr &pass) { PassRegistry::GetInstance()->RegPass(pos, pass); }
  ~PassRegistrar() = default;
};

#define REG_PASS(position, pass) static PassRegistrar g_##position##PassReg(position, std::make_shared<pass>());

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_REGISTRY_PASS_REGISTRY_H_
