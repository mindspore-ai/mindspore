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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_UTILS_H_

#include <string>
#include "include/transform/graph_ir/types.h"
#include "utils/phase.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::transform::OptionMap;

std::string ShapesToString(const ShapeArray &shapes);
std::string GetGraphName(const FuncGraphPtr &graph);
OptionMap GetComputeGraphOptions(const ShapeArray &input_shapes, bool is_dynamic_shape);
bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map, bool export_air);
bool IsGeTrain();
inline std::string GetPhasePrefix() {
  const std::string &phase = PhaseManager::GetInstance().phase();
  auto pos = phase.find('.');
  if (pos != std::string::npos) {
    return phase.substr(0, pos);
  }

  return "";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_UTILS_H_
