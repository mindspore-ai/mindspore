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

#include "plugin/device/cpu/kernel/utils/sampling_kernels.h"
#include <algorithm>
#include <map>

namespace mindspore {
namespace kernel {
static const std::map<std::string, KernelType> kKernelTypeMap = {
  {"lanczos1", Lanczos1}, {"lanczos3", Lanczos3}, {"lanczos5", Lanczos5},   {"gaussian", Gaussian},
  {"box", Box},           {"triangle", Triangle}, {"keyscubic", KeysCubic}, {"mitchellcubic", MitchellCubic}};

KernelType KernelTypeFromString(const std::string &str) {
  auto iter = kKernelTypeMap.find(str);
  if (iter == kKernelTypeMap.end()) {
    return TypeEnd;
  } else {
    return iter->second;
  }
}
}  // namespace kernel
}  // namespace mindspore
