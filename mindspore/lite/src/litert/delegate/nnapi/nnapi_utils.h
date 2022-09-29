/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_UTILS_H_

#include "include/api/types.h"
#include "src/litert/delegate/nnapi/nnapi_implementation.h"

namespace mindspore {
namespace lite {
static const NNAPI *nnapi_ = NNAPIImplementation();

enum PAD {
  PAD_UP = 0,
  PAD_DOWN = 1,
  PAD_LEFT = 2,
  PAD_RIGHT = 3,
};

void ConverTensorQuantSymmToASymm(MSTensor *ms_tensor);
int AddNNAPIOperand(ANeuralNetworksModel *nnapi_model, MSTensor ms_tensor, int idx, int quant_channel_dim = 0,
                    bool is_scalar = false);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_UTILS_H_
