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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_EXIT_ACTOR_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_EXIT_ACTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "src/runtime/lite_mindrt.h"

namespace mindspore::lite {
class LiteExitOpActor : public LiteOpActor {
 public:
  explicit LiteExitOpActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) : LiteOpActor(kernel, ctx) {}
  ~LiteExitOpActor() override = default;
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override;
  int PreInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
              std::unordered_map<Tensor *, Tensor *> *input_map) override;
  int PostInit() override;

 protected:
  void AsyncOutput(OpContext<Tensor> *context) override;
  int PrepareOutputData() override;
  int InitInputData() override;
  int SetInputShape() override;

 private:
  struct MappingInfo {
    MappingInfo(kernel::KernelExec *partial, kernel::KernelExec *call) : partial_node(partial), call_node(call) {}
    kernel::KernelExec *partial_node = nullptr;
    kernel::KernelExec *call_node = nullptr;
    AID partial_input_aid;
    AID call_output_aid;
  };
  int CreateMappingInfo();
  int RecordCallNodeOutputActor(std::vector<std::shared_ptr<LiteOpActor>> *actors);
  void RecordPartialNodeInputActor();
  void SetEntranceInputAID(OpData<Tensor> *inputs);
  bool IsSubSet(const std::vector<lite::Tensor *> &all_set, const std::vector<lite::Tensor *> &sub_set);

  std::vector<std::shared_ptr<LiteOpActor>> *actors_{};
  std::vector<MappingInfo> all_mapping_info_{};
  AID entrance_input_aid_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_EXIT_ACTOR_H_
