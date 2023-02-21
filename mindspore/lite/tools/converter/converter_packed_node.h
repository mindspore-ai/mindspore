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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERT_PACKED_NODE_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERT_PACKED_NODE_H

#include <string>
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
int ConverterPackedNode(schema::MetaGraphT *meta_graph, const std::string &cpu_option);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONVERT_PACKED_NODE_H
