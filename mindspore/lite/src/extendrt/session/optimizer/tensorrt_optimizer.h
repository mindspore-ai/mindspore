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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_OPTIMIZER_H
#define MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_OPTIMIZER_H
#include <vector>

#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "include/api/context.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/lite_graph_executor.h"

namespace mindspore {
class TensorRtOptimizer {
 public:
  void RunOptimizer(const FuncGraphPtr &func_graph);

 private:
  bool OptResizeScales(const FuncGraphPtr &func_graph, const CNodePtr &resize_node);
  bool OptResizeHeightWidth(const FuncGraphPtr &func_graph, const CNodePtr &resize_node);
  tensor::TensorPtr GetParameterValue(const CNodePtr &node, size_t parameter_index);
  std::vector<int32_t> GetParameterIntValue(const CNodePtr &node, size_t parameter_index);
  std::vector<float> GetParameterFloatValue(const CNodePtr &node, size_t parameter_index);
  bool GetMatmulFactor(const AnfNodePtr &pack_input, float *matmul_factor, int32_t *sclice_index,
                       AnfNodePtr *shape_input);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_OPTIMIZER_H
