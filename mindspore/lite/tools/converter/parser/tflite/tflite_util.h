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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_UTIL_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_UTIL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"
#include "schema/schema_generated.h"
#include "schema/inner/ops_generated.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "mindspore/core/utils/check_convert_utils.h"

namespace mindspore {
namespace lite {
std::string GetPadModeStr(tflite::Padding tflite_padmode);

mindspore::PadMode GetPadMode(tflite::Padding tflite_padmode);

size_t GetDataTypeSize(const TypeId &data_type);

mindspore::ActivationType GetActivationFunctionType(tflite::ActivationFunctionType tfliteAFType);

std::string GetMSOpType(tflite::BuiltinOperator tfliteOpType);

TypeId GetTfliteDataType(const tflite::TensorType &tflite_data_type);

STATUS getPaddingParam(const std::unique_ptr<tflite::TensorT> &tensor, mindspore::PadMode pad_mode, int strideH,
                       int strideW, int windowH, int windowW, std::vector<int64_t> *params);

void Split(const std::string &src_str, std::vector<std::string> *dst_str, const std::string &chr);

struct TfliteTensorsInfo {
  std::vector<int32_t> tensorsId;
  std::vector<schema::Format> tensorsFormat;
  std::map<int, int> tensorsIdMap;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_UTIL_H
