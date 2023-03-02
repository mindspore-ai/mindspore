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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMMON_COMPONENT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMMON_COMPONENT_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "src/tensor.h"
#include "tools/converter/micro/coder/context.h"
#include "tools/converter/micro/coder/config.h"

namespace mindspore::lite::micro {
void CodeMSModelCalcWorkspaceSize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                                  const Configurator &config);
void CodeCortexCalcWorkspaceSize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
void CodeMSModelSetWorkspace(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config);
void CodeCortexSetWorkspace(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
void CodeMSTensorHandleArrayDestroyState(std::ofstream &ofs, const Configurator &config);
void CodeMSModelCreateDefault(std::ofstream &ofs);
void CodeMSModelCreate(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config);
void CodeMSModelBuildState(std::ofstream &ofs);
void CodeMSModelBuildCommon(std::ofstream &ofs, const Configurator &config);
void CodeMSModelBuild(std::ofstream &ofs, const int model_index, const Configurator &config);
void CodeMSModelDestory(std::ofstream &ofs, const Configurator *config);
void CodeMSModelPredictState(std::ofstream &ofs);
void CodeMSModelPredictCommon(std::ofstream &ofs);
void CodeMSModelPredict(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config);

void CodeCopyOutputsState(std::ofstream &ofs, const int model_index);
void CodeCopyOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeGlobalCodeBlocks(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeInputState(std::ofstream &ofs, const int model_index);
void CodeInputImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeGraphQuantArgsState(std::ofstream &ofs, const int model_index);
void CodeGraphQuantArgsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeManageResourceState(std::ofstream &ofs, const int model_index);
void CodeInitResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeFreeResourceState(std::ofstream &ofs);
void CodeFreeResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                               const Configurator &config);

void CodeExecuteState(std::ofstream &ofs, const int model_index);
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMMON_COMPONENT_H_
