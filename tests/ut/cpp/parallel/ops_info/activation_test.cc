/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {

class Activation;
class Softmax;
using ActivationPtr = std::shared_ptr<ActivationInfo>;
using SoftmaxPtr = std::shared_ptr<Softmax>;
ActivationPtr act_ptr_;
SoftmaxPtr soft_ptr_;

class TestActivation : public UT::Common {
 public:
  TestActivation() {}
  void SetUp();
  void TearDown() {}
};

void TestActivation::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr relu = MakeValue(std::string("relu"));
  mindspore::HashMap<std::string, ValuePtr> relu_attr = {{"activation_type", relu}};
  ValuePtr sm = MakeValue(std::string("softmax"));
  ValuePtr axix = MakeValue(std::int64_t(2));
  mindspore::HashMap<std::string, ValuePtr> softmax_attr = {{"activation_type", sm}, {"axis", axix}};

  Shapes relu_inputs_shape = {{2, 4, 8, 16}};
  Shapes relu_outputs_shape = {{2, 4, 8, 16}};
  Shapes sm_inputs_shape = {{8, 8, 8, 16}};
  Shapes sm_outputs_shape = {{8, 8, 8, 16}};

  act_ptr_ = std::make_shared<ActivationInfo>("relu_info", relu_inputs_shape, relu_outputs_shape, relu_attr);
  soft_ptr_ = std::make_shared<Softmax>("softmax_info", sm_inputs_shape, sm_outputs_shape, softmax_attr);
}

TEST_F(TestActivation, test_activation_strategies) {
  ASSERT_EQ(act_ptr_->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = act_ptr_->GetStrategyCost();
  for (const auto& swc : sc) {
    ASSERT_NE(swc, nullptr);
    ASSERT_GT(swc->cost_list.size(), 0);
    StrategyPtr sp = swc->strategy_ptr;
    ASSERT_NE(sp, nullptr);
    Cost cost = *(swc->cost_list[0]);

    act_ptr_->InitForCostModel(sp, nullptr);
    std::vector<TensorInfo> inputs_info = act_ptr_->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = act_ptr_->outputs_tensor_info();
    ASSERT_DOUBLE_EQ(act_ptr_->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.computation_cost_);
    ASSERT_DOUBLE_EQ(act_ptr_->operator_cost()->GetCommCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.communication_cost_);
  }
}

TEST_F(TestActivation, DISABLED_test_softmax_strategies) {
  ASSERT_EQ(soft_ptr_->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = soft_ptr_->GetStrategyCost();
  for (const auto& swc : sc) {
    ASSERT_NE(swc, nullptr);
    ASSERT_GT(swc->cost_list.size(), 0);
    StrategyPtr sp = swc->strategy_ptr;
    ASSERT_NE(sp, nullptr);
    Cost cost = *(swc->cost_list[0]);

    Strategies stra = sp->GetInputDim();
    ASSERT_GT(stra.size(), 0);
    Dimensions input0_stra = stra[0];
    ASSERT_GT(input0_stra.size(), 2);
    ASSERT_EQ(input0_stra[2], 1);
    soft_ptr_->InitForCostModel(sp, nullptr);
    std::vector<TensorInfo> inputs_info = soft_ptr_->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = soft_ptr_->outputs_tensor_info();
    ASSERT_DOUBLE_EQ(soft_ptr_->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.computation_cost_);
    ASSERT_DOUBLE_EQ(soft_ptr_->operator_cost()->GetCommCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.communication_cost_);
  }
}

}  // namespace parallel
}  // namespace mindspore
