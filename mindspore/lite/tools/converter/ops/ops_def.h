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

#ifndef LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_OPS_DEF_H_
#define LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_OPS_DEF_H_
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {

enum ConverterPrimitiveType {
  ConverterPrimitiveType_Enter = schema::PrimitiveType_MAX + 1,
  ConverterPrimitiveType_LoopCond,
  ConverterPrimitiveType_NextIteration,
  ConverterPrimitiveType_Exit,
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_NEXTITERATION_H_
