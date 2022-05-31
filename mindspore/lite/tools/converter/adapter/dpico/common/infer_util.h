/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_INFER_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_INFER_UTIL_H_

#include <string>
#include <vector>
#include "include/api/types.h"
#include "schema/model_generated.h"
#include "src/graph_split_info.h"

namespace mindspore {
namespace dpico {
int CheckCustomInputOutput(const std::vector<mindspore::MSTensor> *inputs,
                           const std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive);
int CheckCustomParam(const schema::Custom *param, const std::string &param_name);
int GetOmNetType(const schema::Primitive *primitive, OmNetType *om_net_type);
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_INFER_UTIL_H_
