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

#include <vector>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/redistribution_layout_transfer.h"
#include "util_layout_gen_test.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TestRedistributionLayoutTransfer : public UT::Common {
 public:
  TestRedistributionLayoutTransfer() {}

  void SetUp() { UT::InitPythonPath(); }

  virtual void TearDown() {}
};

void RedistributionLayoutTransferTestFunction(
  const DeviceArrangement &in_device_arrangement_shape, const TensorMap &in_tensor_map_shape,
  const TensorShape &tensor_shape_shape, const DeviceArrangement &out_device_arrangement_shape,
  const TensorMap &out_tensor_map_shape, DeviceArrangement *unified_device_arrangement_shape,
  TensorMap *unified_in_tensor_map_shape, TensorMap *unified_out_tensor_map_shape,
  TensorMap *unified_tensor_shape_shape) {
  Arrangement in_device_arrangement;
  Status status = in_device_arrangement.Init(in_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map in_tensor_map;
  status = in_tensor_map.Init(in_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape;
  status = tensor_shape.Init(tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement out_device_arrangement;
  status = out_device_arrangement.Init(out_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map out_tensor_map;
  status = out_tensor_map.Init(out_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout in_tensor_layout;
  status = in_tensor_layout.Init(in_device_arrangement, in_tensor_map, tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout out_tensor_layout;
  status = out_tensor_layout.Init(out_device_arrangement, out_tensor_map, tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  RedistributionLayoutTransfer tensor_redistribution;
  status = tensor_redistribution.Init(in_tensor_layout, out_tensor_layout);
  ASSERT_EQ(Status::SUCCESS, status);
  std::shared_ptr<ReshapeLayoutTransfer> unified_ptr = tensor_redistribution.UnifyDeviceArrangementAndTensorShape();
  ASSERT_NE(nullptr, unified_ptr);
  TensorLayout unified_in = unified_ptr->from_in();
  TensorLayout unified_out = unified_ptr->to_in();
  // get unified_in result
  Arrangement unified_in_device_arrangement = unified_in.device_arrangement();
  Map unified_in_tensor_map = unified_in.tensor_map();
  *unified_in_tensor_map_shape = unified_in_tensor_map.array();
  Arrangement unified_in_tensor_shape = unified_in.tensor_shape();
  ASSERT_EQ(Status::SUCCESS, status);
  // get unified_out result
  Arrangement unified_out_device_arrangement = unified_out.device_arrangement();
  Map unified_out_tensor_map = unified_out.tensor_map();
  *unified_out_tensor_map_shape = unified_out_tensor_map.array();
  Arrangement unified_out_tensor_shape = unified_out.tensor_shape();
  // checkout valid
  ASSERT_EQ(unified_in_device_arrangement.array(), unified_out_device_arrangement.array());
  ASSERT_EQ(unified_in_tensor_shape.array(), unified_out_tensor_shape.array());
  *unified_device_arrangement_shape = unified_in_device_arrangement.array();
  *unified_tensor_shape_shape = unified_in_tensor_shape.array();
}

void RedistributionLayoutCheck(const DeviceArrangement &in_device_arrangement, const TensorMap &in_tensor_map,
                               const TensorShape &tensor_shape, const DeviceArrangement &out_device_arrangement,
                               const TensorMap &out_tensor_map,
                               const DeviceArrangement &unified_device_arrangement_expect,
                               const TensorMap &unified_in_tensor_map_expect,
                               const TensorMap &unified_out_tensor_map_expect,
                               const TensorMap &unified_tensor_shape_expect) {
  DeviceArrangement unified_device_arrangement;
  TensorMap unified_in_tensor_map;
  TensorMap unified_out_tensor_map;
  TensorShape unified_tensor_shape;
  RedistributionLayoutTransferTestFunction(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement,
                                           out_tensor_map, &unified_device_arrangement, &unified_in_tensor_map,
                                           &unified_out_tensor_map, &unified_tensor_shape);
  // check unified_in result
  ASSERT_EQ(unified_device_arrangement_expect, unified_device_arrangement);
  ASSERT_EQ(unified_in_tensor_map_expect, unified_in_tensor_map);
  ASSERT_EQ(unified_tensor_shape_expect, unified_tensor_shape);
  // check unified_out result
  ASSERT_EQ(unified_device_arrangement_expect, unified_device_arrangement);
  ASSERT_EQ(unified_out_tensor_map_expect, unified_out_tensor_map);
  ASSERT_EQ(unified_tensor_shape_expect, unified_tensor_shape);
}
/*
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [2, 16]
 *    out_tensor_map = [1, 0],
 *    out_tensor_shape = [512, 1024]
 *    in_step1_layout_.device_arrangement = [2, 4, 4]
 *    in_step1_layout_.tensor_map = [2, 1, 0]
 *    in_step1_layout_.tensor_shape = [2, 256, 1024]
 *    out_step1_layout_.device_arrangement = [2, 4, 4]
 *    out_step1_layout_.tensor_map = [2, 1, 0]
 *    out_step1_layout_.tensor_shape = [512, 4, 256]
 *    in_step2_layout_.device_arrangement = [2, 4, 4]
 *    in_step2_layout_.tensor_map = [2, 1, 0, -1]
 *    in_step2_layout_.tensor_shape = [2, 256, 4, 256]
 *    out_step2_layout_.device_arrangement = [2, 4, 4]
 *    out_step2_layout_.tensor_map = [2, -1, 1, 0]
 *    out_step2_layout_.tensor_shape = [2, 256, 4, 256]
 */
TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransfer1) {
  DeviceArrangement in_device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, 0};
  TensorShape tensor_shape = {512, 1024};
  DeviceArrangement out_device_arrangement = {2, 16};
  TensorMap out_tensor_map = {1, 0};
  DeviceArrangement unified_device_arrangement_expect = {2, 4, 4};
  TensorMap unified_in_tensor_map_expect = {2, 1, 0, -1};
  TensorMap unified_out_tensor_map_expect = {2, -1, 1, 0};
  TensorShape unified_tensor_shape_expect = {2, 256, 4, 256};
  RedistributionLayoutCheck(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement, out_tensor_map,
                            unified_device_arrangement_expect, unified_in_tensor_map_expect,
                            unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

/*
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [2, 16]
 *    out_tensor_map = [0, 1],
 *    out_tensor_shape = [512, 1024]
 *    in_step1_layout_.device_arrangement = [2, 4, 4]
 *    in_step1_layout_.tensor_map = [2, 1, 0]
 *    in_step1_layout_.tensor_shape = [2, 256, 1024]
 *    out_step1_layout_.device_arrangement = [2, 4, 4]
 *    out_step1_layout_.tensor_map = [1, 0, 2]
 *    out_step1_layout_.tensor_shape = [4, 128, 1024]
 *    in_step2_layout_.device_arrangement = [2, 2, 2, 4]
 *    in_step2_layout_.tensor_map = [3, 2, 1, 0]
 *    in_step2_layout_.tensor_shape = [2, 2, 128, 1024]
 *    out_step2_layout_.device_arrangement = [2, 2, 2, 4]
 *    out_step2_layout_.tensor_map = [2, 1, 0, 3]
 *    out_step2_layout_.tensor_shape = [2, 2, 128, 1024]
 */
TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransfer2) {
  DeviceArrangement in_device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, 0};
  TensorShape tensor_shape = {512, 1024};
  DeviceArrangement out_device_arrangement = {2, 16};
  TensorMap out_tensor_map = {0, 1};
  DeviceArrangement unified_device_arrangement_expect = {2, 2, 2, 4};
  TensorMap unified_in_tensor_map_expect = {3, 2, 1, 0};
  TensorMap unified_out_tensor_map_expect = {2, 1, 0, 3};
  TensorShape unified_tensor_shape_expect = {2, 2, 128, 1024};
  RedistributionLayoutCheck(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement, out_tensor_map,
                            unified_device_arrangement_expect, unified_in_tensor_map_expect,
                            unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransfer3) {
  DeviceArrangement in_device_arrangement = {8};
  TensorMap in_tensor_map = {0, -1};
  TensorShape tensor_shape = {512, 1024};
  DeviceArrangement out_device_arrangement = {2, 2, 2};
  TensorMap out_tensor_map = {2, 1};
  DeviceArrangement unified_device_arrangement_expect = {2, 2, 2};
  TensorMap unified_in_tensor_map_expect = {2, 1, 0, -1};
  TensorMap unified_out_tensor_map_expect = {2, -1, -1, 1};
  TensorShape unified_tensor_shape_expect = {2, 2, 128, 1024};
  RedistributionLayoutCheck(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement, out_tensor_map,
                            unified_device_arrangement_expect, unified_in_tensor_map_expect,
                            unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransfer4) {
  DeviceArrangement in_device_arrangement = {16, 1, 1};
  TensorMap in_tensor_map = {2, 0};
  TensorShape tensor_shape = {512, 1024};
  DeviceArrangement out_device_arrangement = {1, 16, 1};
  TensorMap out_tensor_map = {2, 1};
  DeviceArrangement unified_device_arrangement_expect = {16};
  TensorMap unified_in_tensor_map_expect = {0, -1};
  TensorMap unified_out_tensor_map_expect = {-1, 0};
  TensorShape unified_tensor_shape_expect = {512, 1024};
  RedistributionLayoutCheck(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement, out_tensor_map,
                            unified_device_arrangement_expect, unified_in_tensor_map_expect,
                            unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransfer5) {
  DeviceArrangement in_device_arrangement = {16};
  TensorMap in_tensor_map = {0, -1};
  TensorShape tensor_shape = {512, 1024};
  DeviceArrangement out_device_arrangement = {16};
  TensorMap out_tensor_map = {-1, 0};
  DeviceArrangement unified_device_arrangement_expect = {16};
  TensorMap unified_in_tensor_map_expect = {0, -1};
  TensorMap unified_out_tensor_map_expect = {-1, 0};
  TensorShape unified_tensor_shape_expect = {512, 1024};
  RedistributionLayoutCheck(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement, out_tensor_map,
                            unified_device_arrangement_expect, unified_in_tensor_map_expect,
                            unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

void ValidRedistributionLayoutCheck(const DeviceArrangement &in_device_arrangement, const TensorMap &in_tensor_map,
                                    const TensorShape &tensor_shape, const DeviceArrangement &out_device_arrangement,
                                    const TensorMap &out_tensor_map) {
  DeviceArrangement unified_device_arrangement;
  TensorMap unified_in_tensor_map;
  TensorMap unified_out_tensor_map;
  TensorShape unified_tensor_shape;
  RedistributionLayoutTransferTestFunction(in_device_arrangement, in_tensor_map, tensor_shape, out_device_arrangement,
                                           out_tensor_map, &unified_device_arrangement, &unified_in_tensor_map,
                                           &unified_out_tensor_map, &unified_tensor_shape);
  // check unified_in result
  ValidLayoutChangeCheck(in_device_arrangement, in_tensor_map, tensor_shape, unified_device_arrangement,
                         unified_in_tensor_map, unified_tensor_shape);
  // check unified_out result
  ValidLayoutChangeCheck(out_device_arrangement, out_tensor_map, tensor_shape, unified_device_arrangement,
                         unified_out_tensor_map, unified_tensor_shape);
}

void ValidRedistributionLayoutCheckAll(int64_t device_pow_size, int64_t tensor_pow_size, int64_t max_device_dim,
                                       int64_t max_shape_dim) {
  std::vector<std::tuple<DeviceArrangement, TensorMap, TensorShape>> layout_list;
  GenerateValidLayoutByDeviceSizeAndTensorSize(device_pow_size, tensor_pow_size, max_device_dim, max_shape_dim,
                                               &layout_list);
  for (size_t in = 0; in < layout_list.size(); in++) {
    for (size_t out = 0; out < layout_list.size(); out++) {
      DeviceArrangement in_device_arrangement = std::get<0>(layout_list[in]);
      TensorMap in_tensor_map = std::get<1>(layout_list[in]);
      TensorShape in_tensor_shape = std::get<2>(layout_list[in]);
      DeviceArrangement out_device_arrangement = std::get<0>(layout_list[out]);
      TensorMap out_tensor_map = std::get<1>(layout_list[out]);
      TensorShape out_tensor_shape = std::get<2>(layout_list[out]);
      if (in_tensor_shape != out_tensor_shape) {
        continue;
      }

      ValidRedistributionLayoutCheck(in_device_arrangement, in_tensor_map, in_tensor_shape, out_device_arrangement,
                                     out_tensor_map);
    }
    if (in % 200 == 0) {
      MS_LOG(INFO) << "sample:" << in << " of " << layout_list.size();
    }
  }
  return;
}

TEST_F(TestRedistributionLayoutTransfer, RedistributionLayoutTransferCheckAll) {
  int64_t device_pow_size_max = 4;
  int64_t tensor_pow_size_max = 4;
  int64_t device_pow_size_min = 1;
  int64_t tensor_pow_size_min = 1;
  const int64_t max_device_dim = 5;
  const int64_t max_shape_dim = 5;
  int64_t device_pow_size = device_pow_size_min;
  while (device_pow_size <= device_pow_size_max) {
    int64_t tensor_pow_size = tensor_pow_size_min;
    while (tensor_pow_size <= tensor_pow_size_max) {
      ValidRedistributionLayoutCheckAll(device_pow_size, tensor_pow_size, max_device_dim, max_shape_dim);
      tensor_pow_size++;
    }
    device_pow_size++;
  }
}

}  // namespace parallel
}  // namespace mindspore
