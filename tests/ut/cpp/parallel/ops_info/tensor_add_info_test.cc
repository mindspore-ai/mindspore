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

#include <string>
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/arithmetic_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class AddInfo;
using AddInfoPtr = std::shared_ptr<AddInfo>;
AddInfoPtr tensor_add, tensor_add1;

class TestTensorAddInfo : public UT::Common {
 public:
  TestTensorAddInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestTensorAddInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 34; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(32);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  std::unordered_map<std::string, ValuePtr> attr;

  Shapes inputs_shape = {{32, 64, 96}, {32, 64, 96}};
  Shapes outputs_shape = {{32, 64, 96}};
  tensor_add = std::make_shared<AddInfo>("tensoradd_info", inputs_shape, outputs_shape, attr);

  Shapes inputs_shape1 = {{1, 48}, {48, 1}};
  Shapes outputs_shape1 = {{48, 48}};
  tensor_add1 = std::make_shared<AddInfo>("tensoradd_info", inputs_shape1, outputs_shape1, attr);
}

TEST_F(TestTensorAddInfo, InferDevMatrixShape1) {
  Strategys inputs = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tensor_add->Init(strategy);
  Shape dev_matrix_shape = tensor_add->dev_matrix_shape();

  Shape expect = {2, 4, 4};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestTensorAddInfo, InferSliceShape1) {
  Strategys str = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, str);

  tensor_add->Init(strategy);
  std::vector<TensorInfo> inputs = tensor_add->inputs_tensor_info();
  std::vector<TensorInfo> outputs = tensor_add->outputs_tensor_info();

  Shape input_slice_shape_expect = {16, 16, 24};
  Shape output_slice_shape_expect = {16, 16, 24};

  TensorInfo inputa_tensor_info = inputs.at(0);
  TensorInfo inputb_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape inputa_slice_shape = inputa_tensor_info.slice_shape();
  Shape inputb_slice_shape = inputb_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(inputa_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(inputb_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestTensorAddInfo, GetTensorLayout1) {
  Strategys str = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, str);

  tensor_add->Init(strategy);
  std::vector<TensorInfo> inputs = tensor_add->inputs_tensor_info();
  std::vector<TensorInfo> outputs = tensor_add->outputs_tensor_info();

  TensorMap input_expect = {2, 1, 0};
  TensorMap output_expect = {2, 1, 0};

  TensorInfo inputa_tensor_info = inputs.at(0);
  TensorInfo inputb_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map inputa_tensor_map = inputa_tensor_info.tensor_layout().origin_tensor_map();
  Map inputb_tensor_map = inputb_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(inputa_tensor_map.array(), input_expect);
  ASSERT_EQ(inputb_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestTensorAddInfo, GetForwardOp1) {
  Strategys inputs = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tensor_add->Init(strategy);
  OperatorVector forward_op = tensor_add->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTensorAddInfo, GetMirrorOPs1) {
  Strategys inputs = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tensor_add->Init(strategy);
  MirrorOps mirror_ops = tensor_add->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestTensorAddInfo, CheckStrategy1) {
  Strategys inputs = {{2, 4, 4}, {2, 6, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tensor_add->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTensorAddInfo, CheckStrategy2) {
  Strategys inputs = {{2, 4, 8}, {2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tensor_add->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTensorAddInfo, CheckStrategy3) {
  Strategys inputs = {{2, 4, 6}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tensor_add->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestTensorAddInfo, CheckStrategy4) {
  Strategys inputs = {{2, 4, 4}, {2, 4, 4}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = tensor_add->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestTensorAddInfo, GenerateStrategies) {
  ASSERT_EQ(tensor_add->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = tensor_add->GetStrategyCost();
  for (auto& swc : sc) {
    StrategyPtr sp = swc->strategy_ptr;
    Cost cost = *(swc->cost_list[0]);
    tensor_add->InitForCostModel(sp);
    std::vector<TensorInfo> inputs_info = tensor_add->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = tensor_add->outputs_tensor_info();
    double memory_cost0 = tensor_add->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage());
    double memory_cost1 = cost.computation_cost_;
    bool memory = memory_cost0 - memory_cost1 <= 1.0;

    double comm_cost0 = tensor_add->operator_cost()->GetCommCost(inputs_info, outputs_info, sp->GetInputStage());
    double comm_cost1 = cost.communication_cost_;
    bool comm = comm_cost0 - comm_cost1 <= 1.0;

    ASSERT_EQ(memory, true);
    ASSERT_EQ(comm, true);
  }
}

TEST_F(TestTensorAddInfo, GenerateStrategies1) {
  ASSERT_EQ(tensor_add1->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = tensor_add1->GetStrategyCost();
  for (auto& swc : sc) {
    StrategyPtr sp = swc->strategy_ptr;
    Cost cost = *(swc->cost_list[0]);
    tensor_add1->InitForCostModel(sp);
    std::vector<TensorInfo> inputs_info = tensor_add1->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = tensor_add1->outputs_tensor_info();
    double memory_cost0 = tensor_add1->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage());
    double memory_cost1 = cost.computation_cost_;
    bool memory = memory_cost0 - memory_cost1 <= 1.0;

    double comm_cost0 = tensor_add1->operator_cost()->GetCommCost(inputs_info, outputs_info, sp->GetInputStage());
    double comm_cost1 = cost.communication_cost_;
    bool comm = comm_cost0 - comm_cost1 <= 1.0;

    ASSERT_EQ(memory, true);
    ASSERT_EQ(comm, true);
  }
}

TEST_F(TestTensorAddInfo, mirror_ops) {
  Strategys inputs = {{1, 8}, {4, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  tensor_add1->Init(strategy);
  MirrorOps mirror_ops = tensor_add1->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}
}  // namespace parallel
}  // namespace mindspore
