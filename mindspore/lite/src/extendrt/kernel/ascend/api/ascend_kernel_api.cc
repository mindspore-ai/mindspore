/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/api/ascend_kernel_api.h"

constexpr auto kNameCustomAscend = "CustomAscend";

std::map<std::string, CreatorFunc> *CreateCustomAscendKernel() {
  CreatorFunc creator_func = []() { return std::make_shared<mindspore::kernel::acl::CustomAscendKernelMod>(); };
  std::map<std::string, CreatorFunc> *func_map = new (std::nothrow) std::map<std::string, CreatorFunc>();
  if (func_map == nullptr) {
    MS_LOG(ERROR) << "New custom ascend kernel failed.";
    return {};
  }
  (*func_map)[kNameCustomAscend] = creator_func;
  return func_map;
}

void DestroyCustomAscendKernel(std::map<std::string, CreatorFunc> *creator_func) {
  if (creator_func == nullptr) {
    MS_LOG(ERROR) << "Param creator func is nullptr.";
    return;
  }
  delete creator_func;
}
