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

#include "coder/user_registry/user_kernel_register.h"
#include <set>

namespace mindspore::lite::micro {
UserKernelFactory *UserKernelFactory::GetInstance() {
  static UserKernelFactory reg;
  return &reg;
}

int UserKernelFactory::RegistUserKernel(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                                        const std::string &header, const std::string &func, const std::string &lib) {
  CoderKey key(target, data_type, operator_type);
  if (user_kernel_sets_.find(key) != user_kernel_sets_.end()) {
    MS_LOG(ERROR) << "Register the user kernel multiple times: " << key.ToString();
    return RET_ERROR;
  }
  user_kernel_sets_[key] = {header, func, lib};
  return RET_OK;
}

std::vector<std::string> UserKernelFactory::FindUserKernel(const CoderKey &key) {
  if (user_kernel_sets_.find(key) != user_kernel_sets_.end()) {
    return user_kernel_sets_[key];
  } else {
    return std::vector<std::string>{};
  }
}

std::set<std::string> UserKernelFactory::UserKernelLibNames() {
  std::set<std::string> names;
  for (auto it = user_kernel_sets_.begin(); it != user_kernel_sets_.end(); ++it) {
    names.insert(it->second[2]);
  }
  return names;
}

UserKernelRegister::UserKernelRegister(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                                       const std::string &header, const std::string &func, const std::string &lib) {
  UserKernelFactory::GetInstance()->RegistUserKernel(target, data_type, operator_type, header, func, lib);
}
}  // namespace mindspore::lite::micro
