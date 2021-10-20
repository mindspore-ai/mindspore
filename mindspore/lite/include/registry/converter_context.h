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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_PARSER_CONTEXT_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_PARSER_CONTEXT_H_

#include <map>
#include <string>
#include "include/lite_utils.h"

namespace mindspore {
namespace converter {
/// \brief FmkType defined frameworks which converter tool supports.
enum MS_API FmkType : int {
  kFmkTypeTf = 0,
  kFmkTypeCaffe = 1,
  kFmkTypeOnnx = 2,
  kFmkTypeMs = 3,
  kFmkTypeTflite = 4,
};

/// \brief ConverterParameters defined read-only converter parameters used by users in ModelParser.
struct MS_API ConverterParameters {
  FmkType fmk;
  std::string model_file;
  std::string weight_file;
  std::map<std::string, std::string> attrs;
};
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_PARSER_CONTEXT_H_
