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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_SWITCH_ACTOR_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_SWITCH_ACTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <set>
#include <utility>
#include "src/litert/lite_mindrt.h"

namespace mindspore::lite {
class LiteSwitchOpActor : public LiteOpActor {
 public:
  explicit LiteSwitchOpActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) : LiteOpActor(kernel, ctx) {}
  ~LiteSwitchOpActor() override {
    delete call_node_;
    delete switch_type_node_;
    for (auto &partial_node : partial_nodes_) {
      delete partial_node;
    }
  };
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override;
  int CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map) override;
  int PrepareOutputData() override;
  std::set<kernel::KernelExec *> GetPartialKernels() const override {
    std::set<kernel::KernelExec *> ret{};
    for (auto &item : partial_nodes_) {
      ret.insert(item);
    }
    return ret;
  }

 protected:
  int UpdateActorOutput() override;

 private:
  STATUS AsyncBranchOutput(const size_t &index, OpContext<Tensor> *context);
  void DecreaseOtherBranchInputTensor(const size_t &index);
  int GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel);
  void AppendOutputTensors();
  int CompileArrowThroughSwitchCall(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map);
  int CreateSwitchTypeArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
                            const std::set<void *> &receiver_tensors, const Tensor *partial_in_tensor,
                            std::vector<DataArrowPtr> *branch_output_data_arrows);
  int ModifySubgraphKernel();
  int SetSwitchPartialNodes();
  int SetSwitchLayerPartialNodes();

  // each element is a set of data arrow sent to the next target actor.
  std::vector<std::vector<DataArrowPtr>> all_branch_output_data_arrows_;

  std::vector<kernel::KernelExec *> partial_nodes_{};
  kernel::KernelExec *switch_type_node_ = nullptr;
  kernel::KernelExec *call_node_ = nullptr;

  // each element is a set of output data which is going to be send to the next target actor.
  std::vector<std::vector<OpDataPtr<Tensor>>> all_branchs_output_data_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_ACTOR_SWITCH_ACTOR_H_
