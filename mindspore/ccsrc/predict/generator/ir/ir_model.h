/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_MODEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_MODEL_H_
#include <string>
#include <vector>
#include <memory>
#include "predict/generator/ir/ir_task_info.h"
namespace mindspore {
namespace generator {
class IRModel {
 public:
  void SetIrTaskInfos(const std::vector<IRtaskInfoPtr> &ir_tasks);
  IRModel() = default;
  ~IRModel();

 private:
  std::vector<IRtaskInfoPtr> ir_tasks_;
};
using IrModelPtr = std::shared_ptr<IRModel>;
}  // namespace generator
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_EXECUTOR_GENERATOR_IR_IR_MODEL_H_
