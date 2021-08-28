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
#ifndef MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_GEN_H_
#define MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_GEN_H_
#include <string>
#include "tools/common/flag_parser.h"

namespace mindspore::lite {
class SchemaGenFlags : public virtual FlagParser {
 public:
  SchemaGenFlags() { AddFlag(&SchemaGenFlags::export_path_, "exportPath", "schema define export path", "."); }
  ~SchemaGenFlags() override = default;

 public:
  std::string export_path_ = ".";
};

class SchemaGen {
 public:
  explicit SchemaGen(SchemaGenFlags *flags) : flags_(flags) {}
  ~SchemaGen() = default;
  int Init();

 private:
  SchemaGenFlags *flags_ = nullptr;
};

int RunSchemaGen(int argc, const char **argv);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_SCHEMA_GEN_SCHEMA_GEN_H_
