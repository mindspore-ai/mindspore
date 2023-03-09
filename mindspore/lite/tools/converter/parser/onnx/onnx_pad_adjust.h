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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_PAD_ADJUST_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_PAD_ADJUST_H_
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::lite {
class OnnxPadAdjust {
 public:
  static bool Adjust(const FuncGraphPtr &func_graph);
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_PAD_ADJUST_H_
