/**
 * Copyright  2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PLUGIN_DEVICE_CPU_OPTIMIZER_CPU_PASS_UTILS_H_
#define MINDSPORE_PLUGIN_DEVICE_CPU_OPTIMIZER_CPU_PASS_UTILS_H_
#include <string>
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const abstract::BaseShapePtr &origin_shape);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_PLUGIN_DEVICE_CPU_OPTIMIZER_CPU_PASS_UTILS_H_
