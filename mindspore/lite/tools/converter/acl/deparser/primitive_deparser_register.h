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

#ifndef ACL_DEPARSER_PRIMITIVE_DEPARSER_REGISTER_H
#define ACL_DEPARSER_PRIMITIVE_DEPARSER_REGISTER_H

#include <map>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "tools/converter/acl/deparser/primitive_deparser.h"

namespace mindspore {
namespace lite {
class PrimitiveDeparserRegister {
 public:
  static PrimitiveDeparserRegister &GetInstance();

  void InsertPrimitiveDeparser(const std::string &name, const PrimitiveDeparserPtr &deparser);

  PrimitiveDeparserPtr GetPrimitiveDeparser(const std::string &name);

 private:
  PrimitiveDeparserRegister() = default;
  ~PrimitiveDeparserRegister() = default;

  std::map<std::string, PrimitiveDeparserPtr> deparser_;
};

class RegisterPrimitiveDeparser {
 public:
  RegisterPrimitiveDeparser(const std::string &name, const PrimitiveDeparserPtr &deparser);

  ~RegisterPrimitiveDeparser() = default;
};

#define REGISTER_PRIMITIVE_DEPARSER(name, deparser) \
  static RegisterPrimitiveDeparser g_##name##PrimDeparser(name, std::make_shared<deparser>());
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_DEPARSER_PRIMITIVE_DEPARSER_REGISTER_H
