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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_ENTRANCE_ACTOR_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_ENTRANCE_ACTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "src/runtime/lite_mindrt.h"

namespace mindspore::lite {
class LiteEntranceOpActor : public LiteOpActor {
 public:
  explicit LiteEntranceOpActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) : LiteOpActor(kernel, ctx) {}
  ~LiteEntranceOpActor() override = default;
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override;

 protected:
  void AsyncOutput(OpContext<Tensor> *context) override;
  int PrepareOutputData() override;
  int InitInputData() override;
  int SetInputShape() override;

 private:
  // record which actor send data
  std::unordered_map<AID, std::vector<OpData<Tensor> *>> input_actor_id_data_{};
  OpDataPtr<Tensor> to_exit_acotr_data_ = nullptr;
  AID entrance_input_aid_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_ENTRANCE_ACTOR_H_
