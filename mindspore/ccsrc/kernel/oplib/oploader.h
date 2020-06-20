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

#ifndef MINDSPORE_OPLOADER_H
#define MINDSPORE_OPLOADER_H

#include <vector>
#include "kernel/oplib/oplib.h"

namespace mindspore {
namespace kernel {
class OpInfoLoaderPy {
 public:
  OpInfoLoaderPy() = default;

  ~OpInfoLoaderPy() = default;

  size_t GetAllOpsInfo() {
    auto ops = OpLib::GetAllOpsInfo();
    auto op_infos = new std::vector<OpInfo *>();
    for (auto op_info : ops) {
      auto new_op_info = new OpInfo(*op_info);
      op_infos->emplace_back(new_op_info);
    }
    return (size_t)op_infos;
  }
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_OPLOADER_H
