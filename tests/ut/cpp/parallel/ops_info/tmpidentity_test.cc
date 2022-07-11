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

#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TmpIdentityInfo;
using TmpIdentityInfoPtr = std::shared_ptr<TmpIdentityInfo>;
TmpIdentityInfoPtr identity_ptr;

class TestTmpIdentityInfo : public UT::Common {
 public:
  TestTmpIdentityInfo() { identity_ptr2 = nullptr; }
  void SetUp();
  void TearDown() {}

  TmpIdentityInfoPtr identity_ptr2;
};

void TestTmpIdentityInfo::SetUp() {
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

  mindspore::HashMap<std::string, ValuePtr> attr = {};
  Shapes inputs_shape = {{2, 4, 8, 16}};
  Shapes outputs_shape = {{2, 4, 8, 16}};
  identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);

  Shapes inputs_shape2 = {{4, 16, 8, 16}};
  Shapes outputs_shape2 = {{4, 16, 8, 16}};
  identity_ptr2 = std::make_shared<TmpIdentityInfo>(inputs_shape2, outputs_shape2, attr);
}

TEST_F(TestTmpIdentityInfo, InferDevMatrixShape1) {
  Strategies inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  identity_ptr->Init(strategy, nullptr);
  Shape dev_matrix_shape = identity_ptr->dev_matrix_shape();

  Shape expect = {2, 4, 8, 16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestTmpIdentityInfo, InferSliceShape1) {
  Strategies str = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  identity_ptr->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = identity_ptr->inputs_tensor_info();
  std::vector<TensorInfo> outputs = identity_ptr->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 1, 1};
  Shape output_slice_shape_expect = {1, 1, 1, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestTmpIdentityInfo, GetTensorLayout1) {
  Strategies str = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  identity_ptr->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = identity_ptr->inputs_tensor_info();
  std::vector<TensorInfo> outputs = identity_ptr->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestTmpIdentityInfo, CheckStrategy1) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = identity_ptr->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTmpIdentityInfo, CheckStrategy2) {
  // Success: {{2,4,8,16}}
  Strategies inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = identity_ptr->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTmpIdentityInfo, test_generate_strategies) {
  ASSERT_EQ(identity_ptr->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = identity_ptr->GetStrategyCost();
  for (const auto& swc : sc) {
    StrategyPtr sp = swc->strategy_ptr;
    Cost cost = *(swc->cost_list[0]);

    identity_ptr->Init(sp, nullptr);
    std::vector<TensorInfo> inputs_info = identity_ptr->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = identity_ptr->outputs_tensor_info();
    ASSERT_DOUBLE_EQ(identity_ptr->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.computation_cost_);
    ASSERT_DOUBLE_EQ(identity_ptr->operator_cost()->GetCommCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.communication_cost_);
  }
}

TEST_F(TestTmpIdentityInfo, test_generate_strategies_base) {
  ASSERT_EQ(identity_ptr->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = identity_ptr->GetStrategyCost();

  Shapes splittable_inputs = {{1, 1, 1, 1}};
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape = {{2, 4, 8, 16}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape, splittable_inputs, &sp_vector);
  ASSERT_EQ(sc.size(), sp_vector.size());
}

TEST_F(TestTmpIdentityInfo, test_generate_strategies_base2) {
  ASSERT_EQ(identity_ptr2->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = identity_ptr2->GetStrategyCost();

  Shapes splittable_inputs = {{1, 1, 1, 1}};
  std::vector<StrategyPtr> sp_vector;
  Shapes inputs_shape2 = {{4, 16, 8, 16}};
  GenerateStrategiesForIndependentInputs(0, inputs_shape2, splittable_inputs, &sp_vector);
  ASSERT_EQ(sc.size(), sp_vector.size());
}
}  // namespace parallel
}  // namespace mindspore
