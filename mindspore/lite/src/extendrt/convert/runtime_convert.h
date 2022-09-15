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
#include "include/api/context.h"
#include "mindapi/ir/func_graph.h"

#ifdef __cplusplus
extern "C" {
#endif
mindspore::api::FuncGraphPtr RuntimeConvert(const char *model_buf, const size_t &buf_size,
                                            const std::shared_ptr<mindspore::Context> &context);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CONVERT_RUNTIME_CONVERT_H_
