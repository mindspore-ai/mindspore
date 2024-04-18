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
#include "graph/ascend_string.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
namespace ge {
AscendString::AscendString(char const *name) {}

Operator::Operator(const string &name, const string &type) {}
Operator::Operator(const AscendString &name, const AscendString &type) {}
Operator::Operator(const char *name, const char *type) {}
Operator::Operator(const std::string &type) {}

std::string Operator::GetName() const { return ""; }

void Operator::InputRegister(const std::string &name) {}
void Operator::InputRegister(const char *name) {}

void Operator::OutputRegister(const std::string &name) {}
void Operator::OutputRegister(const char *name) {}

void Operator::OptionalInputRegister(const std::string &name) {}
void Operator::OptionalInputRegister(const char *name) {}

void Operator::DynamicInputRegister(const std::string &name, const uint32_t num, bool is_push_back) {}
void Operator::DynamicInputRegister(const char *name, const uint32_t num, bool is_push_back) {}
void Operator::DynamicOutputRegister(const std::string &name, const uint32_t num, bool is_push_back) {}
void Operator::DynamicOutputRegister(const char *name, const uint32_t num, bool is_push_back) {}

void Operator::AttrRegister(const std::string &name, int64_t attr_value) {}
void Operator::AttrRegister(const char *name, int64_t attr_value) {}
void Operator::RequiredAttrRegister(const std::string &name) {}
void Operator::RequiredAttrRegister(const char *name) {}
void Operator::RequiredAttrWithTypeRegister(const char_t *name, const char_t *type) {}

OperatorCreatorRegister::OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator) {}
OperatorCreatorRegister::OperatorCreatorRegister(char const *,
                                                 std::function<ge::Operator(ge::AscendString const &)> const &) {}
}  // namespace ge