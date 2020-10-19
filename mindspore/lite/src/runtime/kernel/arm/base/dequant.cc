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
#include "src/runtime/kernel/arm/base/dequant.h"

namespace mindspore::kernel {
float *DequantUtil::DequantWeight(lite::Tensor *input_tensor) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->data_type() != kNumberTypeInt8 && input_tensor->data_type() != kNumberTypeInt16) {
    MS_LOG(ERROR) << "Conv weight input type error." << input_tensor->data_type();
    return nullptr;
  }
  if (input_tensor->GetQuantParams().empty()) {
    MS_LOG(ERROR) << "No quant param.";
    return nullptr;
  }
  if (input_tensor->data_type() == kNumberTypeInt16) {
    return DequantData<int16_t>(input_tensor);
  } else {
    return DequantData<int8_t>(input_tensor);
  }
}
}  // namespace mindspore::kernel
