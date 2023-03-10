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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INSERT_TENSOR_MOVE_FOR_COMMUNICATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INSERT_TENSOR_MOVE_FOR_COMMUNICATION_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
// If the input Tensor of the graph is connected to the AllReduce operator,
// and the input Tensor of the graph already has a device address,
// we need to copy the data in the device address to the contiguous memory of AllReduce.
class BACKEND_EXPORT InsertTensorMoveForCommunication : public Pass {
 public:
  InsertTensorMoveForCommunication() : Pass("insert_tensor_move_for_communication") {}
  ~InsertTensorMoveForCommunication() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INSERT_TENSOR_MOVE_FOR_COMMUNICATION_H_
