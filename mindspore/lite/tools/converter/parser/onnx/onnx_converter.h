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

#ifndef MS_ONNX_CONVERTER_H
#define MS_ONNX_CONVERTER_H
#include <string>
#include <memory>
#include "tools/converter/converter.h"
#include "tools/converter/graphdef_transform.h"

namespace mindspore {
namespace lite {
class OnnxConverter : public Converter {
 public:
  OnnxConverter();

  ~OnnxConverter() override = default;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MS_ONNX_CONVERTER_H

