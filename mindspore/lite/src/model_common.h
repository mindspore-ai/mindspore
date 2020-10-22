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

#ifndef MINDSPORE_LITE_SRC_MODEL_COMMON_H_
#define MINDSPORE_LITE_SRC_MODEL_COMMON_H_
#include "src/ops/primitive_c.h"
#include "include/model.h"

namespace mindspore::lite {
bool ConvertNodes(const schema::MetaGraph *meta_graph, Model *model);

bool ConvertTensors(const schema::MetaGraph *meta_graph, Model *model);

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_MODEL_COMMON_H_
