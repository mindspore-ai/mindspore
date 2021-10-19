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

#include "include/registry/model_parser_registry.h"
#include <map>
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace registry {
namespace {
std::map<FmkType, ModelParserCreator> model_parser_room;
}  // namespace

ModelParserRegistry::ModelParserRegistry(FmkType fmk, ModelParserCreator creator) {
  if (fmk < converter::kFmkTypeTf || fmk > converter::kFmkTypeTflite) {
    MS_LOG(ERROR) << "ILLEGAL FMK: fmk must be in FmkType.";
    return;
  }
  model_parser_room[fmk] = creator;
}

converter::ModelParser *ModelParserRegistry::GetModelParser(FmkType fmk) {
  if (fmk < converter::kFmkTypeTf || fmk > converter::kFmkTypeTflite) {
    MS_LOG(ERROR) << "ILLEGAL FMK: fmk must be in FmkType.";
    return nullptr;
  }
  auto it = model_parser_room.find(fmk);
  if (it != model_parser_room.end()) {
    auto creator = it->second;
    MS_CHECK_TRUE_RET(creator != nullptr, nullptr);
    return creator();
  }
  return nullptr;
}
}  // namespace registry
}  // namespace mindspore
