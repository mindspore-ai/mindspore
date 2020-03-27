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
#include "parallel/strategy.h"
#include "parallel/ops_info/get_next_info.h"
#include "parallel/device_manager.h"
#include "parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class GetNextInfo;
using GetNextInfoPtr = std::shared_ptr<GetNextInfo>;
GetNextInfoPtr get_next;

class TestGetNextInfo : public UT::Common {
 public:
  TestGetNextInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestGetNextInfo::SetUp() {
  std::list<int32_t> dev_list;

  for (int32_t i = 0; i < 8; i++) {
    dev_list.push_back(i);
  }

  std::list<int32_t> stage_map;
  stage_map.push_back(8);
  int32_t local_dev = 0;
  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
  Shapes inputs_shape = {};
  Shapes outputs_shape = {{64, 32}, {64}};
  std::unordered_map<std::string, ValuePtr> attr;
  std::vector<std::string> types_ = {"float32", "int32"};
  Shapes shapes_ = {{64, 32}, {64}};
  int32_t output_num_ = 2;
  std::string shared_name_ = "test_get_next";
  attr["types"] = MakeValue(types_);
  attr["shapes"] = MakeValue(shapes_);
  attr["output_num"] = MakeValue(output_num_);
  attr["shared_name"] = MakeValue(shared_name_);
  get_next = std::make_shared<GetNextInfo>("get_next_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestGetNextInfo, InferDevMatrixShape1) {
  std::vector<Dimensions> inputs = {{}, {}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  get_next->Init(strategy);
  std::vector<int32_t> dev_matrix_shape = get_next->dev_matrix_shape();
  std::vector<int32_t> expect = {8, 1};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestGetNextInfo, InferSliceShape1) {
  std::vector<Dimensions> str = {{}, {}};
  StrategyPtr strategy = NewStrategy(0, str);

  get_next->Init(strategy);
  std::vector<TensorInfo> outputs = get_next->outputs_tensor_info();
  Shape output_slice_shape_expect0 = {8, 32};
  Shape output_slice_shape_expect1 = {8};
  TensorInfo output_tensor_info0 = outputs.at(0);
  TensorInfo output_tensor_info1 = outputs.at(1);
  Shape output_slice_shape0 = output_tensor_info0.slice_shape();
  Shape output_slice_shape1 = output_tensor_info1.slice_shape();
  ASSERT_EQ(output_slice_shape0, output_slice_shape_expect0);
  ASSERT_EQ(output_slice_shape1, output_slice_shape_expect1);
}

TEST_F(TestGetNextInfo, GetTensorLayout1) {
  std::vector<Dimensions> str = {{}, {}};
  StrategyPtr strategy = NewStrategy(0, str);
  get_next->Init(strategy);
  std::vector<TensorInfo> outputs = get_next->outputs_tensor_info();
  TensorMap output_expect0 = {1, 0};
  TensorMap output_expect1 = {1};
  TensorInfo output_tensor_info0 = outputs.at(0);
  TensorInfo output_tensor_info1 = outputs.at(1);

  Map output_tensor_map0 = output_tensor_info0.tensor_layout().origin_tensor_map();
  Map output_tensor_map1 = output_tensor_info1.tensor_layout().origin_tensor_map();
  ASSERT_EQ(output_tensor_map0.array(), output_expect0);
  ASSERT_EQ(output_tensor_map1.array(), output_expect1);
}

TEST_F(TestGetNextInfo, CheckStrategy1) {
  std::vector<Dimensions> inputs = {};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = get_next->Init(strategy);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(TestGetNextInfo, CheckStrategy2) {
  std::vector<Dimensions> inputs = {{8, 1}, {8}};
  StrategyPtr strategy = NewStrategy(0, inputs);
  Status ret = get_next->Init(strategy);
  ASSERT_EQ(ret, FAILED);
}
}  // namespace parallel
}  // namespace mindspore
