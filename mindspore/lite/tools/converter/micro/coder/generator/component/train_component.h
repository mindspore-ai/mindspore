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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_TRAIN_COMPONENT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_TRAIN_COMPONENT_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "src/tensor.h"
#include "tools/converter/micro/coder/context.h"

namespace mindspore::lite::micro {
void CodeMSModelSetTrainModeState(std::ofstream &ofs);
void CodeMSModelSetTrainModeCommon(std::ofstream &ofs);
void CodeMSModelSetTrainMode(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
void CodeMSModelRunStepState(std::ofstream &ofs);
void CodeMSModelRunStepCommon(std::ofstream &ofs);
void CodeMSModelRunStep(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
void CodeMSModelExportWeightState(std::ofstream &ofs);
void CodeMSModelExportWeightCommon(std::ofstream &ofs);
void CodeMSModelExportWeight(std::ofstream &ofs, const int model_index);
void CodeWeightInitFuncForTrain(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
void CodeCopyTrainOutputsState(std::ofstream &ofs, const int model_index);
void CodeCopyTrainOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_TRAIN_COMPONENT_H_
