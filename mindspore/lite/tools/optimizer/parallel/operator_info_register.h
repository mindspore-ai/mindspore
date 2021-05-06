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

#ifndef MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H
#define MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "tools/optimizer/parallel/operator_info.h"

namespace mindspore {
namespace opt {
using OperatorInfoCreatorFunc =
  std::function<std::unique_ptr<opt::OperatorInfo>(const std::string &name, const SplitStrategy &strategy)>;

class OperatorInfoFactory {
 public:
  static OperatorInfoFactory *GeInstance();

  OperatorInfoFactory(const OperatorInfoFactory &) = delete;

  OperatorInfoFactory &operator=(const OperatorInfoFactory &) = delete;

  int RegisterOperatorInfo(const std::string &name, const SplitStrategy &strategy,
                           const OperatorInfoCreatorFunc &creator_func);

  OperatorInfoCreatorFunc FindOperatorInfo(const std::string &name, const SplitStrategy &strategy);

 private:
  OperatorInfoFactory() = default;

  virtual ~OperatorInfoFactory() = default;

 private:
  std::map<std::string, OperatorInfoCreatorFunc> operator_info_map_;
};

class OperatorInfoRegister {
 public:
  OperatorInfoRegister() = delete;

  OperatorInfoRegister(const std::string &name, const SplitStrategy &strategy,
                       const OperatorInfoCreatorFunc &creator_func);

  ~OperatorInfoRegister() = default;
};

#define OPERATOR_INFO_REGISTER(name, strategy, creator_func) \
  static OperatorInfoRegister g_##name##Creator(name, strategy, creator_func);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H
