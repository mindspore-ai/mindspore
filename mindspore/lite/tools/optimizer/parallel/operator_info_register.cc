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

#include "tools/optimizer/parallel/operator_info_register.h"
#include <utility>
namespace mindspore {
namespace opt {

OperatorInfoFactory *OperatorInfoFactory::GeInstance() {
  static OperatorInfoFactory factory;
  return &factory;
}

int OperatorInfoFactory::RegisterOperatorInfo(const std::string &name, const SplitStrategy &strategy,
                                              const OperatorInfoCreatorFunc &creator_func) {
  if (operator_info_map_.find(name) != operator_info_map_.end()) {
    MS_LOG(ERROR) << "Operator already exist" << name;
    return lite::RET_ERROR;
  }
  this->operator_info_map_.insert(std::pair<std::string, OperatorInfoCreatorFunc>(name, creator_func));
  return lite::RET_OK;
}

OperatorInfoCreatorFunc OperatorInfoFactory::FindOperatorInfo(const std::string &name, const SplitStrategy &strategy) {
  auto iterator = this->operator_info_map_.find(name);
  if (iterator != this->operator_info_map_.end()) {
    return iterator->second;
  }
  MS_LOG(ERROR) << "operator_info do not register" << name;
  return nullptr;
}

OperatorInfoRegister::OperatorInfoRegister(const std::string &name, const SplitStrategy &strategy,
                                           const OperatorInfoCreatorFunc &creator_func) {
  OperatorInfoFactory::GeInstance()->RegisterOperatorInfo(name, strategy, creator_func);
}
}  // namespace opt
}  // namespace mindspore
