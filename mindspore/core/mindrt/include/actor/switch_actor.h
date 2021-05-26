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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_SWITCH_ACTOR_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_SWITCH_ACTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include "actor/actor.h"
#include "actor/op_actor.h"

namespace mindspore {

template <typename T>
class SwitchActorBase : public OpActor<T> {
 public:
  explicit SwitchActorBase(std::string op_name) : OpActor<T>(op_name) {}
  virtual ~SwitchActorBase() = default;

  // The actor run when receive the input data.
  void RunOpData(OpData<T> *input_data, OpContext<T> *context = nullptr) override {}

 protected:
  // Different output branches according to the input.
  std::vector<std::vector<DataArrowPtr>> output_branch_arrows_;
};

}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_SWITCH_ACTOR_H
