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
#include <algorithm>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/reshape_layout_transfer.h"
#include "util_layout_gen_test.h"
#include "utils/log_adapter.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class TestReshapeLayoutTransfer : public UT::Common {
 public:
  TestReshapeLayoutTransfer() {}

  void SetUp() { UT::InitPythonPath(); }

  virtual void TearDown() {}
};

void InferUnifiedLayout(const DeviceArrangement &device_arrangement_shape, const TensorMap &in_tensor_map_shape,
                        const TensorShape &in_tensor_shape_shape, const TensorMap &out_tensor_map_shape,
                        const TensorShape &out_tensor_shape_shape, DeviceArrangement *unified_device_arrangement_shape,
                        TensorMap *unified_in_tensor_map_shape, TensorMap *unified_out_tensor_map_shape,
                        TensorMap *unified_tensor_shape_shape) {
  Arrangement device_arrangement;
  Status status = device_arrangement.Init(device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map in_tensor_map;
  status = in_tensor_map.Init(in_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement in_tensor_shape;
  status = in_tensor_shape.Init(in_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout in_tensor_layout;
  status = in_tensor_layout.Init(device_arrangement, in_tensor_map, in_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map out_tensor_map;
  status = out_tensor_map.Init(out_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement out_tensor_shape;
  status = out_tensor_shape.Init(out_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout out_tensor_layout;
  status = out_tensor_layout.Init(device_arrangement, out_tensor_map, out_tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  ReshapeLayoutTransfer tensor_redistribution;
  status = tensor_redistribution.Init(in_tensor_layout, out_tensor_layout);
  ASSERT_EQ(Status::SUCCESS, status);
  std::shared_ptr<ReshapeLayoutTransfer> unified_ptr = tensor_redistribution.UnifyDeviceArrangementAndTensorShape();
  ASSERT_NE(nullptr, unified_ptr);
  TensorLayout unified_in = unified_ptr->from_in();
  TensorLayout unified_out = unified_ptr->to_in();
  // get unified_in result
  Arrangement unified_in_device_arrangement = unified_in.device_arrangement();
  Arrangement unified_in_tensor_shape = unified_in.tensor_shape();
  Map unified_in_tensor_map = unified_in.tensor_map();
  // get unified_out result
  Arrangement unified_out_device_arrangement = unified_out.device_arrangement();
  Arrangement unified_out_tensor_shape = unified_out.tensor_shape();
  Map unified_out_tensor_map = unified_out.tensor_map();
  // checkout valid
  ASSERT_EQ(unified_in_device_arrangement.array(), unified_out_device_arrangement.array());
  ASSERT_EQ(unified_in_tensor_shape.array(), unified_out_tensor_shape.array());
  *unified_device_arrangement_shape = unified_in_device_arrangement.array();
  *unified_tensor_shape_shape = unified_in_tensor_shape.array();
  *unified_in_tensor_map_shape = unified_in_tensor_map.array();
  *unified_out_tensor_map_shape = unified_out_tensor_map.array();
}

void InferUnifiedLayoutCheck(const DeviceArrangement &device_arrangement, const TensorMap &in_tensor_map,
                             const TensorShape &in_tensor_shape, const TensorMap &out_tensor_map,
                             const TensorShape &out_tensor_shape,
                             const DeviceArrangement &unified_device_arrangement_expect,
                             const TensorMap &unified_in_tensor_map_expect,
                             const TensorMap &unified_out_tensor_map_expect,
                             const TensorMap &unified_tensor_shape_expect) {
  DeviceArrangement unified_device_arrangement;
  TensorMap unified_in_tensor_map;
  TensorMap unified_out_tensor_map;
  TensorShape unified_tensor_shape;
  InferUnifiedLayout(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape,
                     &unified_device_arrangement, &unified_in_tensor_map, &unified_out_tensor_map,
                     &unified_tensor_shape);
  // check unified_in result
  ASSERT_EQ(unified_device_arrangement_expect, unified_device_arrangement);
  ASSERT_EQ(unified_in_tensor_map_expect, unified_in_tensor_map);
  ASSERT_EQ(unified_tensor_shape_expect, unified_tensor_shape);
  // check unified_out result
  ASSERT_EQ(unified_device_arrangement_expect, unified_device_arrangement);
  ASSERT_EQ(unified_out_tensor_map_expect, unified_out_tensor_map);
  ASSERT_EQ(unified_tensor_shape_expect, unified_tensor_shape);
}

void ValidUnifiedLayoutCheck(const DeviceArrangement &device_arrangement, const TensorMap &in_tensor_map,
                             const TensorShape &in_tensor_shape, const TensorMap &out_tensor_map,
                             const TensorShape &out_tensor_shape) {
  DeviceArrangement unified_device_arrangement;
  TensorMap unified_in_tensor_map;
  TensorMap unified_out_tensor_map;
  TensorShape unified_tensor_shape;
  InferUnifiedLayout(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape,
                     &unified_device_arrangement, &unified_in_tensor_map, &unified_out_tensor_map,
                     &unified_tensor_shape);
  // check unified_in result
  ValidLayoutChangeCheck(device_arrangement, in_tensor_map, in_tensor_shape, unified_device_arrangement,
                         unified_in_tensor_map, unified_tensor_shape);
  // check unified_out result
  ValidLayoutChangeCheck(device_arrangement, out_tensor_map, out_tensor_shape, unified_device_arrangement,
                         unified_out_tensor_map, unified_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, InferUnifiedLayout1) {
  DeviceArrangement device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, 0};
  TensorShape in_tensor_shape = {512, 1024};
  TensorMap out_tensor_map = {1, 0};
  TensorShape out_tensor_shape = {128, 4096};
  DeviceArrangement unified_device_arrangement_expect = {8, 4};
  TensorMap unified_in_tensor_map_expect = {1, -1, 0};
  TensorMap unified_out_tensor_map_expect = {1, 0, -1};
  TensorShape unified_tensor_shape_expect = {128, 4, 1024};
  InferUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape,
                          unified_device_arrangement_expect, unified_in_tensor_map_expect,
                          unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

TEST_F(TestReshapeLayoutTransfer, InferUnifiedLayout2) {
  DeviceArrangement device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, 0};
  TensorShape in_tensor_shape = {512, 1024};
  TensorMap out_tensor_map = {0, 1};
  TensorShape out_tensor_shape = {128, 4096};
  DeviceArrangement unified_device_arrangement_expect = {4, 2, 4};
  TensorMap unified_in_tensor_map_expect = {2, 1, -1, 0};
  TensorMap unified_out_tensor_map_expect = {0, -1, 2, 1};
  TensorShape unified_tensor_shape_expect = {4, 32, 4, 1024};
  InferUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape,
                          unified_device_arrangement_expect, unified_in_tensor_map_expect,
                          unified_out_tensor_map_expect, unified_tensor_shape_expect);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck1) {
  DeviceArrangement device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, 0};
  TensorShape in_tensor_shape = {512, 1024};
  TensorMap out_tensor_map = {0, 1};
  TensorShape out_tensor_shape = {128, 4096};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck2) {
  DeviceArrangement device_arrangement = {8, 4};
  TensorMap in_tensor_map = {1, -1};
  TensorShape in_tensor_shape = {512, 1024};
  TensorMap out_tensor_map = {0, -1};
  TensorShape out_tensor_shape = {128, 4096};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck3) {
  DeviceArrangement device_arrangement = {2};
  TensorMap in_tensor_map = {-1, -1};
  TensorShape in_tensor_shape = {4, 2};
  TensorMap out_tensor_map = {-1, -1};
  TensorShape out_tensor_shape = {2, 4};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck4) {
  DeviceArrangement device_arrangement = {4, 8};
  TensorMap in_tensor_map = {-1, -1, -1};
  TensorShape in_tensor_shape = {8, 2, 2};
  TensorMap out_tensor_map = {-1, -1, 0};
  TensorShape out_tensor_shape = {2, 2, 8};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck5) {
  DeviceArrangement device_arrangement = {2, 2};
  TensorMap in_tensor_map = {0, -1};
  TensorShape in_tensor_shape = {2, 2};
  TensorMap out_tensor_map = {0, -1};
  TensorShape out_tensor_shape = {2, 2};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck6) {
  DeviceArrangement device_arrangement = {2, 2};
  TensorMap in_tensor_map = {0, 1};
  TensorShape in_tensor_shape = {2, 8};
  TensorMap out_tensor_map = {0, 1};
  TensorShape out_tensor_shape = {2, 8};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck7) {
  DeviceArrangement device_arrangement = {2, 4, 4};
  TensorMap in_tensor_map = {-1, 0, -1};
  TensorShape in_tensor_shape = {2, 4, 4};
  TensorMap out_tensor_map = {-1, -1, 0};
  TensorShape out_tensor_shape = {2, 2, 8};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck8) {
  DeviceArrangement device_arrangement = {4, 2, 4};
  TensorMap in_tensor_map = {2, 0, 1};
  TensorShape in_tensor_shape = {4, 4, 2};
  TensorMap out_tensor_map = {-1, -1, -1};
  TensorShape out_tensor_shape = {2, 4, 4};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

// this reshape layout can not be generated by ValidInferUnifiedLayoutCheckAll
TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck9) {
  DeviceArrangement device_arrangement = {2, 2, 2};
  TensorMap in_tensor_map = {2, 1, 0, -1};
  TensorShape in_tensor_shape = {2, 2, 128, 1024};
  TensorMap out_tensor_map = {2, 1};
  TensorShape out_tensor_shape = {512, 1024};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck10) {
  DeviceArrangement device_arrangement = {2, 2, 2};
  TensorMap in_tensor_map = {2, 1, 0, -1};
  TensorShape in_tensor_shape = {2, 2, 2, 2};
  TensorMap out_tensor_map = {2, 1};
  TensorShape out_tensor_shape = {8, 2};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheck11) {
  DeviceArrangement device_arrangement = {2};
  TensorMap in_tensor_map = {0, -1};
  TensorShape in_tensor_shape = {2, 2};
  TensorMap out_tensor_map = {0};
  TensorShape out_tensor_shape = {4};
  ValidUnifiedLayoutCheck(device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
}

void ValidInferUnifiedLayoutCheckAll(int64_t device_pow_size, int64_t tensor_pow_size, int64_t max_device_dim,
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
      if (in_device_arrangement != out_device_arrangement) {
        continue;
      }

      ValidUnifiedLayoutCheck(in_device_arrangement, in_tensor_map, in_tensor_shape, out_tensor_map, out_tensor_shape);
    }
    if (in % 200 == 0) {
      MS_LOG(INFO) << "sample:" << in << " of " << layout_list.size();
    }
  }
  return;
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheckAll) {
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
      ValidInferUnifiedLayoutCheckAll(device_pow_size, tensor_pow_size, max_device_dim, max_shape_dim);
      tensor_pow_size++;
    }
    device_pow_size++;
  }
}

TEST_F(TestReshapeLayoutTransfer, ValidInferUnifiedLayoutCheckAll2) {
  int64_t device_pow_size_max = 1;
  int64_t tensor_pow_size_max = 2;
  int64_t device_pow_size_min = 1;
  int64_t tensor_pow_size_min = 2;
  const int64_t max_device_dim = 5;
  const int64_t max_shape_dim = 5;
  int64_t device_pow_size = device_pow_size_min;
  while (device_pow_size <= device_pow_size_max) {
    int64_t tensor_pow_size = tensor_pow_size_min;
    while (tensor_pow_size <= tensor_pow_size_max) {
      ValidInferUnifiedLayoutCheckAll(device_pow_size, tensor_pow_size, max_device_dim, max_shape_dim);
      tensor_pow_size++;
    }
    device_pow_size++;
  }
}
}  // namespace parallel
}  // namespace mindspore
