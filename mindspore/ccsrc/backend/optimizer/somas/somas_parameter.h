/**
 * Copyright 2021 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_PARAMETER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_PARAMETER_H_

#include <memory>
#include "base/base.h"

namespace mindspore {
namespace somas {
class SomasParameter {
 public:
  SomasParameter(size_t id, AnfNodePtr source_node, size_t index, const void *addr, size_t size)
      : id_(id), source_node_(source_node), output_index_(index), addr_(const_cast<void *>(addr)), size_(size) {}
  ~SomasParameter() = default;

  const size_t id_{0};
  AnfNodePtr source_node_;
  size_t output_index_;
  void *addr_;
  size_t size_;
};
using SomasParameterPtr = std::shared_ptr<SomasParameter>;
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_PARAMETER_H_
