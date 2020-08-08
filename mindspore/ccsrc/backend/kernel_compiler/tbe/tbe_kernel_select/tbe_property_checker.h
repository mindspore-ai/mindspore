/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_SELECT_TBE_PROPERTY_CHECKER_H
#define MINDSPORE_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_SELECT_TBE_PROPERTY_CHECKER_H
#include "mindspore/core/ir/anf.h"

namespace mindspore {
namespace kernel {
class TbePropertyChecker {
 public:
  TbePropertyChecker() = default;
  ~TbePropertyChecker() = default;
  static bool CheckTbeProperties(const mindspore::CNodePtr &cnode);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_SELECT_TBE_PROPERTY_CHECKER_H
