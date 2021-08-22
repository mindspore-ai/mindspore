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
#include <string>
#include <set>
#include <unordered_map>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
ModelParserRegistry *ModelParserRegistry::GetInstance() {
  static ModelParserRegistry instance;
  return &instance;
}

ModelParser *ModelParserRegistry::GetModelParser(const FmkType fmk) {
  auto it = parsers_.find(fmk);
  if (it != parsers_.end()) {
    auto creator = it->second;
    return creator();
  }
  return nullptr;
}

int ModelParserRegistry::RegParser(const FmkType fmk, ModelParserCreator creator) {
  if (fmk < converter::FmkType_TF || fmk > converter::FmkType_TFLITE) {
    MS_LOG(ERROR) << "ILLEGAL FMK: fmk must be in FmkType.";
    return RET_ERROR;
  }
  auto instance = ModelParserRegistry::GetInstance();
  instance->parsers_[fmk] = creator;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
