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

#include <vector>
#include "common/common_test.h"
#include "ir/value.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/construct_operator.h"

namespace mindspore {
namespace parallel {

class MatMulInfo;
using MatMulInfoPtr = std::shared_ptr<MatMulInfo>;
ConstructOperator constructor;

class TestConstructOperator : public UT::Common {
 public:
  TestConstructOperator() {}

  void SetUp();

  virtual void TearDown() {}
};

void TestConstructOperator::SetUp() {
  RankList dev_list;

  for (int64_t i = 0; i < 1050; i++) {
    dev_list.push_back(i);
  }
  RankList stage_map;
  stage_map.push_back(1024);
  stage_map.push_back(26);

  int32_t local_dev = 0;

  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};

  Shapes inputs_shape_1 = {{2, 4, 8, 16}, {2, 4, 16, 32}};
  Shapes outputs_shape_1 = {{2, 4, 8, 32}};

  MatMulInfoPtr matmul = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);

  Strategies str = {{2, 4, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, str);
  matmul->Init(strategy, nullptr);
  Shape tensor_shape = {512, 1024};
  Shape dev_matrix_shape = {2, 4, 8, 16, 1};
  RankList used_dev_list = g_device_manager->GetDeviceListByStageId(0);
  constructor.Init(used_dev_list, dev_matrix_shape, false);
  constructor.UpdateTensorShape(tensor_shape);
}

TEST_F(TestConstructOperator, DISABLED_TestReshapeOP) {
  Shape shape = {512, 512, 2};
  ASSERT_EQ(constructor.ReshapeOP(shape), Status::SUCCESS);
}

TEST_F(TestConstructOperator, DISABLED_TestStridedSliceOP) {
  Args args = {1, 2, 3};
  int64_t split_count = args[0];
  int64_t split_dim = args[1];
  Shape device_arrangement = {8, 4};
  Arrangement dev_mat;
  dev_mat.Init(device_arrangement);
  Shape map = {1, -1};
  Map tensor_map;
  tensor_map.Init(map);
  Shape shape = {512, 1024};
  Arrangement tensor_shape;
  tensor_shape.Init(shape);
  TensorLayout tensor_layout;
  tensor_layout.Init(dev_mat, tensor_map, tensor_shape);
  ASSERT_EQ(constructor.StridedSliceOP(args), Status::SUCCESS);

  Operator op = constructor.GetOperator();
  OperatorParams params = op.second.second;
  ValuePtr begin_ptr = params[0].first.second;
  ValuePtr end_ptr = params[1].first.second;
  Shape begin = GetValue<const std::vector<int64_t>>(begin_ptr);
  Shape end = GetValue<const std::vector<int64_t>>(end_ptr);
  for (size_t i = 0; i < begin.size(); i++) {
    int64_t diff = end[i] - begin[i];
    int64_t num = shape[i];
    if (SizeToLong(i) != split_dim) {
      ASSERT_EQ(diff, shape[i]);
    } else {
      ASSERT_EQ(diff, num / split_count);
    }
  }
}

TEST_F(TestConstructOperator, DISABLED_TestAllGatherOP) {
  int64_t dev_dim = 2;
  ASSERT_EQ(constructor.AllGatherOP(dev_dim), Status::SUCCESS);
}

TEST_F(TestConstructOperator, DISABLED_TestConcatOP) {
  int64_t concat_dim = 0;
  ASSERT_EQ(constructor.ConcatOP(concat_dim), Status::SUCCESS);
}

TEST_F(TestConstructOperator, DISABLED_TestSplitOP) {
  int64_t split_count = 2;
  ASSERT_EQ(constructor.SplitOP(split_count), Status::SUCCESS);
}

TEST_F(TestConstructOperator, DISABLED_TestAlltoAllOP) {
  int64_t split_count = 2;
  int64_t split_dim = 0;
  int64_t concat_dim = 1;
  int64_t dev_dim = 3;
  int64_t dev_num = 8;
  Args args = {split_count, split_dim, concat_dim, dev_dim, dev_num};
  ASSERT_EQ(constructor.AlltoAllOP(args), Status::SUCCESS);
}

}  // namespace parallel
}  // namespace mindspore
