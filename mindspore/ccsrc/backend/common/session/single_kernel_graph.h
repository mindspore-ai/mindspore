/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_
#define MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_

#include <string>
#include <vector>
#include <memory>
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace session {
class BACKEND_EXPORT SingleKernelGraph {
 public:
  SingleKernelGraph() = default;
  ~SingleKernelGraph() = default;

  static std::shared_ptr<session::KernelGraph> ConstructKernelGraphBasedOnSingleOp(
    const std::string &op_name, const std::vector<TypeId> &input_dtypes, const std::vector<ShapeVector> &input_shapes,
    const std::vector<TypeId> &output_dtypes, const std::vector<ShapeVector> &output_shapes);
};
}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_
