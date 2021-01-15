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
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {

class MatMulInfo;
using MatMulInfoPtr = std::shared_ptr<MatMulInfo>;
MatMulInfoPtr matmul1;
MatMulInfoPtr matmul2;
MatMulInfoPtr matmul3;
MatMulInfoPtr matmul4;

class TestMatmulInfo : public UT::Common {
 public:
  TestMatmulInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestMatmulInfo::SetUp() {
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

  // matmul1
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpoce_b_1 = MakeValue(false);
  std::unordered_map<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpoce_b_1}};

  Shapes inputs_shape_1 = {{2, 4, 8, 16}, {2, 4, 16, 32}};
  Shapes outputs_shape_1 = {{2, 4, 8, 32}};

  matmul1 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);

  // matmul2
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpoce_b_2 = MakeValue(true);
  std::unordered_map<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpoce_b_2}};

  Shapes inputs_shape_2 = {{2, 4, 8, 16}, {32, 16}};
  Shapes outputs_shape_2 = {{2, 4, 8, 32}};

  matmul2 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);

  // matmul3
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpoce_b_3 = MakeValue(true);
  std::unordered_map<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpoce_b_3}};

  Shapes inputs_shape_3 = {{8, 16}, {2, 4, 32, 16}};
  Shapes outputs_shape_3 = {{2, 4, 8, 32}};

  matmul3 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);

  // matmul4
  std::unordered_map<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_3}};
  matmul4 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_4);
}

TEST_F(TestMatmulInfo, InferDevMatrixShape1) {
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  Shape dev_matrix_shape = matmul1->dev_matrix_shape();

  Shape expect = {2, 4, 8, 16, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestMatmulInfo, InferDevMatrixShape2) {
  Strategys inputs = {{2, 4, 8, 8}, {2, 4, 8, 2}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  Shape dev_matrix_shape = matmul1->dev_matrix_shape();

  Shape expect = {2, 4, 8, 8, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

// matmul2
TEST_F(TestMatmulInfo, InferDevMatrixShape3) {
  Strategys inputs = {{2, 4, 8, 16}, {1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul2->Init(strategy);
  Shape dev_matrix_shape = matmul2->dev_matrix_shape();

  Shape expect = {2, 4, 8, 16, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

// matmul2
TEST_F(TestMatmulInfo, InferDevMatrixShape4) {
  Strategys inputs = {{2, 4, 8, 8}, {2, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul2->Init(strategy);
  Shape dev_matrix_shape = matmul2->dev_matrix_shape();

  Shape expect = {2, 4, 8, 8, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

// matmul3
TEST_F(TestMatmulInfo, InferDevMatrixShape5) {
  Strategys inputs = {{8, 16}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul3->Init(strategy);
  Shape dev_matrix_shape = matmul3->dev_matrix_shape();

  Shape expect = {2, 4, 8, 16, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

// matmul3
TEST_F(TestMatmulInfo, InferDevMatrixShape6) {
  Strategys inputs = {{8, 8}, {2, 4, 2, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul3->Init(strategy);
  Shape dev_matrix_shape = matmul3->dev_matrix_shape();

  Shape expect = {2, 4, 8, 8, 2};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestMatmulInfo, InferTensorMap1) {
  Strategys str = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul1->Init(strategy);
  std::vector<TensorInfo> inputs = matmul1->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul1->outputs_tensor_info();

  TensorMap mat_a_expect = {4, 3, 2, 1};
  TensorMap mat_b_expect = {4, 3, 1, 0};
  TensorMap output_expect = {4, 3, 2, 0};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map mat_a_tensor_map = mat_a_tensor_info.tensor_layout().origin_tensor_map();
  Map mat_b_tensor_map = mat_b_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(mat_a_tensor_map.array(), mat_a_expect);
  ASSERT_EQ(mat_b_tensor_map.array(), mat_b_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

// matmul2
TEST_F(TestMatmulInfo, InferTensorMap2) {
  Strategys str = {{2, 4, 8, 16}, {1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul2->Init(strategy);
  std::vector<TensorInfo> inputs = matmul2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul2->outputs_tensor_info();

  TensorMap mat_a_expect = {4, 3, 2, 1};
  TensorMap mat_b_expect = {0, 1};
  TensorMap output_expect = {4, 3, 2, 0};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map mat_a_tensor_map = mat_a_tensor_info.tensor_layout().origin_tensor_map();
  Map mat_b_tensor_map = mat_b_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(mat_a_tensor_map.array(), mat_a_expect);
  ASSERT_EQ(mat_b_tensor_map.array(), mat_b_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

// matmul3
TEST_F(TestMatmulInfo, InferTensorMap3) {
  Strategys str = {{8, 16}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul3->Init(strategy);
  std::vector<TensorInfo> inputs = matmul3->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul3->outputs_tensor_info();

  TensorMap mat_a_expect = {2, 1};
  TensorMap mat_b_expect = {4, 3, 0, 1};
  TensorMap output_expect = {4, 3, 2, 0};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map mat_a_tensor_map = mat_a_tensor_info.tensor_layout().origin_tensor_map();
  Map mat_b_tensor_map = mat_b_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(mat_a_tensor_map.array(), mat_a_expect);
  ASSERT_EQ(mat_b_tensor_map.array(), mat_b_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestMatmulInfo, InferSliceShape1) {
  Strategys str = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul1->Init(strategy);
  std::vector<TensorInfo> inputs = matmul1->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul1->outputs_tensor_info();

  Shape mat_a_slice_shape_expect = {1, 1, 1, 1};
  Shape mat_b_slice_shape_expect = {1, 1, 1, 32};
  Shape output_slice_shape_expect = {1, 1, 1, 32};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape mat_a_slice_shape = mat_a_tensor_info.slice_shape();
  Shape mat_b_slice_shape = mat_b_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(mat_a_slice_shape, mat_a_slice_shape_expect);
  ASSERT_EQ(mat_b_slice_shape, mat_b_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

// matmul2
TEST_F(TestMatmulInfo, InferSliceShape2) {
  Strategys str = {{2, 4, 8, 16}, {1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul2->Init(strategy);
  std::vector<TensorInfo> inputs = matmul2->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul2->outputs_tensor_info();

  Shape mat_a_slice_shape_expect = {1, 1, 1, 1};
  Shape mat_b_slice_shape_expect = {32, 1};
  Shape output_slice_shape_expect = {1, 1, 1, 32};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape mat_a_slice_shape = mat_a_tensor_info.slice_shape();
  Shape mat_b_slice_shape = mat_b_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(mat_a_slice_shape, mat_a_slice_shape_expect);
  ASSERT_EQ(mat_b_slice_shape, mat_b_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

// matmul3
TEST_F(TestMatmulInfo, InferSliceShape3) {
  Strategys str = {{8, 16}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul3->Init(strategy);
  std::vector<TensorInfo> inputs = matmul3->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul3->outputs_tensor_info();

  Shape mat_a_slice_shape_expect = {1, 1};
  Shape mat_b_slice_shape_expect = {1, 1, 32, 1};
  Shape output_slice_shape_expect = {1, 1, 1, 32};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape mat_a_slice_shape = mat_a_tensor_info.slice_shape();
  Shape mat_b_slice_shape = mat_b_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(mat_a_slice_shape, mat_a_slice_shape_expect);
  ASSERT_EQ(mat_b_slice_shape, mat_b_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

// matmul3
TEST_F(TestMatmulInfo, GetTensorLayout3) {
  Strategys str = {{8, 16}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  matmul3->Init(strategy);
  std::vector<TensorInfo> inputs = matmul3->inputs_tensor_info();
  std::vector<TensorInfo> outputs = matmul3->outputs_tensor_info();

  TensorMap mat_a_expect = {2, 1};
  TensorMap mat_b_expect = {4, 3, 0, 1};
  TensorMap output_expect = {4, 3, 2, 0};

  TensorInfo mat_a_tensor_info = inputs.at(0);
  TensorInfo mat_b_tensor_info = inputs.at(1);
  TensorInfo output_tensor_info = outputs.at(0);

  Map mat_a_tensor_map = mat_a_tensor_info.tensor_layout().origin_tensor_map();
  Map mat_b_tensor_map = mat_b_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(mat_a_tensor_map.array(), mat_a_expect);
  ASSERT_EQ(mat_b_tensor_map.array(), mat_b_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestMatmulInfo, GetForwardOp1) {
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  OperatorVector forward_op = matmul1->forward_op();

  OperatorArgs operator_args = forward_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;

  std::string arg1_name = operator_args.first.at(1).first;
  ValuePtr arg1_value = operator_args.first.at(1).second;
  bool arg1_value_is_string = false;
  if (arg1_value->isa<StringImm>()) {
    arg1_value_is_string = true;
  }

  ASSERT_EQ(forward_op.at(0).first, "AllReduce");
  ASSERT_EQ(forward_op.size(), 1);
  ASSERT_EQ(arg0_name, "op");
  ASSERT_EQ(arg1_name, "group");
  ASSERT_EQ(arg1_value_is_string, true);
}

TEST_F(TestMatmulInfo, GetForwardOp2) {
  Strategys inputs = {{2, 4, 8, 1}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  OperatorVector forward_op = matmul1->forward_op();

  ASSERT_EQ(forward_op.size(), 0);
}

TEST_F(TestMatmulInfo, GetVirtualDivOp1) {
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  OperatorVector virtual_div_op = matmul1->virtual_div_op();

  OperatorArgs operator_args = virtual_div_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  int64_t divisor = arg0_value->cast<Int64ImmPtr>()->value();

  ASSERT_EQ(virtual_div_op.at(0).first, "_VirtualDiv");
  ASSERT_EQ(virtual_div_op.size(), 1);
  ASSERT_EQ(arg0_name, "divisor");
  ASSERT_EQ(divisor, 16);
}

TEST_F(TestMatmulInfo, GetMirrorOPs1) {
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  MirrorOps mirror_ops = matmul1->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

// matmul2
TEST_F(TestMatmulInfo, GetMirrorOPs2) {
  Strategys inputs = {{2, 4, 1, 16}, {8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul2->Init(strategy);
  MirrorOps mirror_ops = matmul2->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;
  std::string group = arg0_value->cast<StringImmPtr>()->ToString();

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

// matmul3
TEST_F(TestMatmulInfo, GetMirrorOPs3) {
  Strategys inputs = {{8, 16}, {2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul3->Init(strategy);
  MirrorOps mirror_ops = matmul3->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestMatmulInfo, GetMirrorOPs4) {
  Strategys inputs = {{2, 4, 1, 16}, {2, 4, 16, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  matmul1->Init(strategy);
  MirrorOps mirror_ops = matmul1->mirror_ops();

  ASSERT_EQ(mirror_ops.size(), 2);
}

TEST_F(TestMatmulInfo, InitTwice) {
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  // init twice
  matmul1->Init(strategy);
  matmul1->Init(strategy);

  MirrorOps mirror_ops = matmul1->mirror_ops();
  OperatorVector mirror_op = mirror_ops.at(1);

  OperatorArgs operator_args = mirror_op.at(0).second;

  std::string arg0_name = operator_args.first.at(0).first;
  ValuePtr arg0_value = operator_args.first.at(0).second;

  ASSERT_EQ(mirror_op.at(0).first, "_MirrorOperator");
  ASSERT_EQ(mirror_op.size(), 1);
  ASSERT_EQ(arg0_name, "group");
}

TEST_F(TestMatmulInfo, CheckStrategy1) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy2) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{2, 4, 8, 16}, {4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy3) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{2, 4, 8, 16}, {2, 4, 8, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy4) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{2, 4, 8, 16}, {2, 3, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy5) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{0, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy6) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{-1, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, CheckStrategy7) {
  // Success: {{2,4,8,16}, {2,4,16,1}}
  Strategys inputs = {{4, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul1->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, InitFailed) {
  // matmul4 attr is wrong
  Strategys inputs = {{4, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = matmul4->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestMatmulInfo, test_GenerateStrategies1) {
  // the parameter '0' indicates that the stageId = 0, there are 1024 devices in the stage 0
  ASSERT_EQ(matmul1->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = matmul1->GetStrategyCost();
  for (const auto& swc : sc) {
    StrategyPtr sp = swc->strategy_ptr;
    Cost cost = *(swc->cost_list[0]);
    matmul1->InitForCostModel(sp);
    std::vector<TensorInfo> inputs_info = matmul1->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = matmul1->outputs_tensor_info();
    ASSERT_DOUBLE_EQ(matmul1->operator_cost()->GetComputationCost(inputs_info, outputs_info, sp->GetInputStage()),
                     cost.computation_cost_);
    break;
  }
}

TEST_F(TestMatmulInfo, test_GenerateStrategies2) {
  // the parameter '0' indicates that the stageId = 0, there are 1024 devices in the stage 0
  ASSERT_EQ(matmul3->GenerateStrategies(0), Status::SUCCESS);
  std::vector<std::shared_ptr<StrategyWithCost>> sc = matmul3->GetStrategyCost();
  for (const auto& swc : sc) {
    StrategyPtr sp = swc->strategy_ptr;
    Cost cost = *(swc->cost_list[0]);
    matmul3->InitForCostModel(sp);

    std::vector<TensorInfo> inputs_info = matmul3->inputs_tensor_info();
    std::vector<TensorInfo> outputs_info = matmul3->outputs_tensor_info();
    std::vector<TensorInfo> replica_inputs_info;
    replica_inputs_info.push_back(inputs_info[0]);

    // transpose the tensor B
    TensorInfo input1_info = inputs_info[1];
    Shape input1_shape = input1_info.shape();
    Shape input1_slice_shape = input1_info.slice_shape();
    TensorLayout tly;
    matmul3->SwapLastTwoElements(&input1_shape);
    matmul3->SwapLastTwoElements(&input1_slice_shape);
    TensorInfo replica_input1_info(tly, input1_shape, input1_slice_shape);
    replica_inputs_info.push_back(replica_input1_info);

    ASSERT_DOUBLE_EQ(matmul3->operator_cost()->GetComputationCost(replica_inputs_info, outputs_info, sp->GetInputStage()),
                     cost.computation_cost_);
    break;
  }
}
}  // namespace parallel
}  // namespace mindspore
