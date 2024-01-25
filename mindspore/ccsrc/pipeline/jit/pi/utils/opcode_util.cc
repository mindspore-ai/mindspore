/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/utils/opcode_util.h"
#include "pipeline/jit/pi/pydef.h"

namespace mindspore {
namespace pijit {
namespace code {
const Opcode &GetOpcodeInfo(int op) {
  static Opcode kOpcodes[NO_IMPL_OPCODE] = {{"", 0, 0}};
  if (!kOpcodes[0].name_.empty()) {
    return op > 0 && op < NO_IMPL_OPCODE ? kOpcodes[op] : kOpcodes[NO_IMPL_OPCODE - 1];
  }
  kOpcodes[0] = Opcode();
#define DEF_OPCODE_ATTR(code, flag)                                 \
  kOpcodes[code < NO_IMPL_OPCODE ? code : (NO_IMPL_OPCODE - 1)] = { \
    /* define opcode attr, and check opcode define */               \
    (code < NO_IMPL_OPCODE ? #code : "NOT_IMPLEMENT"), flag, code};
#include "opcode_attr.def"
#undef DEF_OPCODE_ATTR
  return kOpcodes[op];
}

#undef DEF_OPCODE_ATTR
}  // namespace code
}  // namespace pijit
}  // namespace mindspore
