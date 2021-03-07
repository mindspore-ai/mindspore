/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "mindspore/lite/tools/converter/quantizer/quantizer.h"

namespace mindspore::lite::quant {

STATUS Quantizer::GenerateQuantParam() { return RET_OK; }

STATUS Quantizer::RemoveFakeQuant() { return RET_OK; }

STATUS Quantizer::DetermineNodeQuantType() { return RET_OK; }

STATUS FbQuantizer::GenerateQuantParam() { return RET_OK; }

STATUS FbQuantizer::RemoveFakeQuant() { return RET_OK; }

STATUS FbQuantizer::DetermineNodeQuantType() { return RET_OK; }
}  // namespace mindspore::lite::quant
