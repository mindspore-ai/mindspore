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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CONVERT_RUNTIME_CONVERT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CONVERT_RUNTIME_CONVERT_H_

#include <stdio.h>
#include <string>
#include <memory>
#include <map>
#include "include/api/context.h"
#include "mindapi/ir/func_graph.h"
#include "src/common/config_infos.h"

#ifdef __cplusplus
extern "C" {
#endif
int RuntimeConvert(const mindspore::api::FuncGraphPtr &graph, const std::shared_ptr<mindspore::Context> &context,
                   const mindspore::ConfigInfos &config_info);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CONVERT_RUNTIME_CONVERT_H_
