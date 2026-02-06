/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_ONNX_ONNX_EXPORTER_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_ONNX_ONNX_EXPORTER_H_

#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "ir/func_graph.h"
#include "include/utils/visible.h"

namespace mindspore {
COMMON_EXPORT std::string GetOnnxProtoString(const FuncGraphPtr &func_graph,
                                             const std::vector<std::string> &input_names = {},
                                             const std::vector<std::string> &outputs_names = {}, int opset_version = 11,
                                             bool export_params = true, bool keep_initializers_as_inputs = false,
                                             const std::map<std::string, std::map<int, std::string>> &dynamic_axes = {},
                                             bool extra_save_params = false, const std::string &save_file_dir = "");
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_ONNX_ONNX_EXPORTER_H_
