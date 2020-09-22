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

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/tensor_layout/redistribution_operator_infer.h"
#include "frontend/parallel/device_manager.h"
#include "util_layout_gen_test.h"

namespace mindspore {
namespace parallel {

class TestRedistributionOperatorInfer : public UT::Common {
 public:
  TestRedistributionOperatorInfer() {}

  void SetUp() {
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
  }

  virtual void TearDown() {}
};

// check if in_tensor_map could be changed to out_tensor_map with operator_list
void InferOperatorCheck(Shape in_tensor_map, const Shape &out_tensor_map, const OperatorList &operator_list) {
  for (auto op_cost : operator_list) {
    OperatorR op = op_cost.first;
    Args args = op.second;
    ASSERT_GT(args.size(), 2);
    std::string str = op.first;
    if (str == SPLIT_BY_AXIS) {
      in_tensor_map[args[1]] = out_tensor_map[args[1]];
    } else if (str == PERMUTE_BY_AXIS) {
      in_tensor_map[args[1]] = in_tensor_map[args[2]];
      in_tensor_map[args[2]] = -1;
    } else {
      in_tensor_map[args[0]] = -1;
    }
  }
  ASSERT_EQ(in_tensor_map, out_tensor_map);
}

// generate all the valid tensor_map with the length of dim
Status InferOperatorCheckAll(uint32_t dim_len) {
  Shapes tensor_map_list;
  Shape dev_mat;
  for (uint32_t i = 0; i < dim_len; i++) dev_mat.push_back(2);
  Shape tensor_shape = dev_mat;
  GenerateValidTensorMap(dev_mat, tensor_shape, &tensor_map_list);
  RankList dev_list;
  for (int32_t i = 0; i < static_cast<int32_t>(pow(2, dim_len)); i++) dev_list.push_back(i);
  Arrangement in_dev_mat;
  in_dev_mat.Init(dev_mat);
  Arrangement in_tensor_shape;
  in_tensor_shape.Init(tensor_shape);
  for (auto iter1 = tensor_map_list.begin(); iter1 < tensor_map_list.end(); iter1++) {
    for (auto iter2 = tensor_map_list.begin(); iter2 < tensor_map_list.end(); iter2++) {
      if (iter1 == iter2) continue;
      TensorLayout layout;
      Map in_tensor_map;
      in_tensor_map.Init(*iter1);
      layout.Init(in_dev_mat, in_tensor_map, in_tensor_shape);
      RedistributionOperatorInfer operatorInfer;
      Map out_tensor_map;
      out_tensor_map.Init(*iter2);
      if (operatorInfer.Init(layout, out_tensor_map, dev_list) == Status::FAILED) {
        return Status::FAILED;
      }
      operatorInfer.InferRedistributionOperator();
      OperatorList operator_list = operatorInfer.operator_list();
      InferOperatorCheck(*iter1, *iter2, operator_list);
    }
  }
  return Status::SUCCESS;
}

TEST_F(TestRedistributionOperatorInfer, TestInferOperatorAll) {
  uint32_t dim_len = 3;
  Status status = InferOperatorCheckAll(dim_len);
  ASSERT_EQ(status, Status::SUCCESS);
}

}  // namespace parallel
}  // namespace mindspore
