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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PARAMETER_TO_TENSOR_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PARAMETER_TO_TENSOR_H_
#include "ir/func_graph.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
class ParameterToTensor : public opt::Pass {
 public:
  ParameterToTensor() : Pass("parameter_to_tensor") {}
  ~ParameterToTensor() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PARAMETER_TO_TENSOR_H_
