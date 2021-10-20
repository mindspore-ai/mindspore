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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_REGISTRY_H
#define MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_REGISTRY_H

#include <memory>
#include "include/lite_utils.h"
#include "include/registry/converter_context.h"

using mindspore::converter::FmkType;
namespace mindspore {
namespace converter {
class ModelParser;
}  // namespace converter
namespace registry {
/// \brief ModelParserCreator defined function pointer to get a ModelParser class.
typedef converter::ModelParser *(*ModelParserCreator)();

/// \brief ModelParserRegistry defined registration and storage of ModelParser.
class MS_API ModelParserRegistry {
 public:
  /// \brief Constructor of ModelParserRegistry.
  ///
  /// \param[in] fmk Define identification of a certain framework.
  /// \param[in] creator Define function pointer of creating ModelParser.
  ModelParserRegistry(FmkType fmk, ModelParserCreator creator);

  /// \brief Destructor of ModelParserRegistry.
  ~ModelParserRegistry() = default;

  /// \brief Static Method to get a model parser.
  ///
  /// \param[in] fmk Define identification of a certain framework.
  ///
  /// \return Pointer of ModelParser.
  static converter::ModelParser *GetModelParser(FmkType fmk);
};

/// \brief Defined registering macro to register ModelParser, which called by user directly.
///
/// \param[in] fmk Define identification of a certain framework.
/// \param[in] parserCreator Define function pointer of creating ModelParser.
#define REG_MODEL_PARSER(fmk, parserCreator) \
  static mindspore::registry::ModelParserRegistry g_##type##fmk##ModelParserReg(fmk, parserCreator);
}  // namespace registry
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_REGISTRY_H
