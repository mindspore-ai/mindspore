/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_
#define MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_

#include <memory>

namespace mindspore {
enum Axis : int {
  N = 0,
  C,
  H,
  W,
};
// Interface for device kernel program information.
class KernelInfoDevice {
 public:
  // If kernel program was built and build info is set.
  virtual bool has_build_info() const = 0;
};
using KernelInfoDevicePtr = std::shared_ptr<KernelInfoDevice>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_
