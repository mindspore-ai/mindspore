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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMPONENT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMPONENT_H_
#include <string>

namespace mindspore::lite::micro {
extern const char *kInputPrefixName;
extern const char *kOutputPrefixName;
extern const char *kWeightPrefixName;
constexpr auto kPackWeightOffsetName = "w_offset";
constexpr auto kPackWeightSizeName = "w_size";
extern const char *kBufferPrefixName;
extern const char *kBufferPrefixNameAdd;
void FreeGlobalVariable();
void InitGlobalVariable(int model_index);

constexpr auto kModelName = "net";

constexpr auto kSourcePath = "/src/";

constexpr auto kBenchmarkPath = "/benchmark/";
constexpr auto kBenchmarkFile = "benchmark.cc";

constexpr auto kSession = "session";
constexpr auto kTensor = "tensor";

constexpr auto kNameSpaceMindSpore = "namespace mindspore";
constexpr auto kNameSpaceLite = "namespace lite";

constexpr auto kAlignedString = "  ";

constexpr auto kDebugUtils = "debug_utils.h";

constexpr auto kThreadWrapper = "wrapper/thread/micro_thread_pool.h";

constexpr auto kExternCpp =
  "#ifdef __cplusplus\n"
  "extern \"C\" {\n"
  "#endif\n";

constexpr auto kEndExternCpp =
  "#ifdef __cplusplus\n"
  "}\n"
  "#endif\n";
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_GENERATOR_COMPONENT_COMPONENT_H_
