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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_ALLOCATOR_COMPONENT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_ALLOCATOR_COMPONENT_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "src/tensor.h"
#include "tools/converter/micro/coder/config.h"
#include "tools/converter/micro/coder/context.h"

namespace mindspore::lite::micro {
void CodeAllocatorFileHeader(std::ofstream &ofs);
void CodeCalcRefCount(std::ofstream &ofs);
void CodeGlobalMemory(std::ofstream &ofs, size_t size);
void CodeMemoryOccupied(std::ofstream &ofs);
void CodeLockOccupied(std::ofstream &ofs);
}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_ALLOCATOR_COMPONENT_H_
