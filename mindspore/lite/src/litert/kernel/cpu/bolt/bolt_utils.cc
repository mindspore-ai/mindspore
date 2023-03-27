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

#include "bolt/bolt_utils.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel::bolt {
int ConvertActType(const ActType &lite_act, ActivationMode *bolt_act) {
  switch (lite_act) {
    case ActType_No:
      *bolt_act = ACTIVATION_NULL;
      break;
    case ActType_Relu:
      *bolt_act = ACTIVATION_RELU;
      break;
    case ActType_Sigmoid:
      *bolt_act = ACTIVATION_SIGMOID;
      break;
    case ActType_Relu6:
      *bolt_act = ACTIVATION_RELU6;
      break;
    case ActType_Abs:
      *bolt_act = ACTIVATION_ABS;
      break;
    case ActType_Softplus:
      *bolt_act = ACTIVATION_SOFTPLUS;
      break;
    case ActType_Tanh:
      *bolt_act = ACTIVATION_TANH;
      break;
    case ActType_HSwish:
      *bolt_act = ACTIVATION_H_SWISH;
      break;
    case ActType_HSigmoid:
      *bolt_act = ACTIVATION_H_SIGMOID;
      break;
    case ActType_Sign:
      *bolt_act = ACTIVATION_SIGN;
      break;
    case ActType_Swish:
      *bolt_act = ACTIVATION_SWISH;
      break;
    case ActType_Gelu:
      *bolt_act = ACTIVATION_GELU;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported act type: " << lite_act << " for bolt";
      return RET_NOT_SUPPORT;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel::bolt
