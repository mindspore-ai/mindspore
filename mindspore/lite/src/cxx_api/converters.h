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

#ifndef MINDSPORE_LITE_SRC_CXX_API_CONVERTERS_H_
#define MINDSPORE_LITE_SRC_CXX_API_CONVERTERS_H_

#include <limits.h>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/lite_types.h"
#include "src/cxx_api/context.h"

namespace mindspore {

namespace lite {
struct Context;
class TrainCfg;
}  // namespace lite

class Context;
class TrainCfg;

inline lite::QuantizationType A2L_ConvertQT(mindspore::QuantizationType qt) {
  if (qt == kNoQuant) {
    return lite::QT_NONE;
  }
  if (qt == kWeightQuant) {
    return lite::QT_WEIGHT;
  }
  return lite::QT_DEFAULT;
}

inline lite::CpuBindMode A2L_ConvertAffinityMode(int affinity_mode) {
  switch (affinity_mode) {
    case 0:
      return lite::NO_BIND;
    case 1:
      return lite::HIGHER_CPU;
    case 2:
      return lite::MID_CPU;
    default:
      return lite::NO_BIND;
  }
}

inline bool IsAffinityModeValid(int affinity_mode) {
  return affinity_mode >= lite::NO_BIND && affinity_mode <= lite::MID_CPU;
}

Status A2L_ConvertContext(Context *a_context, lite::Context *l_context);
Status A2L_ConvertContext(const Context::Data *a_context, lite::Context *l_context);
Status A2L_ConvertConfig(const TrainCfg *a_train_cfg, lite::TrainCfg *l_train_cfg);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_CONVERTERS_H_
