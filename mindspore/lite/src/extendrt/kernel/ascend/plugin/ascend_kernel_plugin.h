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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_KERNEL_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_KERNEL_PLUGIN_H_

#include <map>
#include <string>
#include <memory>
#include "kernel/kernel.h"

namespace mindspore::kernel {
using KernelModFunc = std::function<std::shared_ptr<kernel::KernelMod>()>;

class AscendKernelPlugin {
 public:
  static AscendKernelPlugin &GetInstance();

  void Register();
  void UpdateRegisterStatus(bool status);
  void DestroyAscendKernelMap();

 private:
  AscendKernelPlugin();
  ~AscendKernelPlugin();

  void *handle_;
  std::map<std::string, KernelModFunc> *create_kernel_map_;
  bool is_registered_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_KERNEL_PLUGIN_H_
