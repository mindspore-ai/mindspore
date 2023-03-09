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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_INSERT_FORMAT_TRANSFORM_OP_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_INSERT_FORMAT_TRANSFORM_OP_H

#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "ir/anf.h"

namespace mindspore {
namespace opt {
class InsertFormatTransformOpCPU : public Pass {
 public:
  explicit InsertFormatTransformOpCPU(const std::string &) : Pass("insert_format_transform_op_cpu") {}
  ~InsertFormatTransformOpCPU() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_INSERT_FORMAT_TRANSFORM_OP_H
