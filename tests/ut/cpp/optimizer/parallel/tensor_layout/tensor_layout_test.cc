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
#include "optimizer/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {

class TestTensorLayout : public UT::Common {
 public:
  TestTensorLayout() {}

  void SetUp() { UT::InitPythonPath(); }

  virtual void TearDown() {}
};

void ReshapeExpandDeviceArrangementTestFunction(const std::vector<int32_t>& in_device_arrangement_shape,
                                                const std::vector<int32_t>& in_tensor_map_shape,
                                                const std::vector<int32_t>& in_tensor_shape_shape,
                                                const std::vector<int32_t>& out_device_arrangement_shape,
                                                const std::vector<int32_t>& out_tensor_map_shape,
                                                const std::vector<int32_t>& out_tensor_shape_shape) {
  Arrangement device_arrangement;
  Status status = device_arrangement.Init(in_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map tensor_map;
  status = tensor_map.Init(in_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape;
  status = tensor_shape.Init(in_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement device_arrangement_new;
  status = device_arrangement_new.Init(out_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout tensor_layout;
  tensor_layout.Init(device_arrangement, tensor_map, tensor_shape);
  std::shared_ptr<TensorLayout> tensor_layout_new_ptr = tensor_layout.ExpandDeviceArrangement(device_arrangement_new);
  ASSERT_NE(nullptr, tensor_layout_new_ptr);
  Map tensor_map_new = tensor_layout_new_ptr->tensor_map();
  ASSERT_EQ(out_tensor_map_shape, tensor_map_new.array());
  Arrangement tensor_shape_new = tensor_layout_new_ptr->tensor_shape();
  ASSERT_EQ(out_tensor_shape_shape, tensor_shape_new.array());
}

/*
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [3, 2, 1, 0],
 *    out_tensor_shape = [4, 128, 2, 512]
 *
 */
TEST_F(TestTensorLayout, ReshapeExpandDeviceArrangement1) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {1, 0};
  std::vector<int32_t> tensor_shape = {512, 1024};
  std::vector<int32_t> device_arrangement_new = {4, 2, 2, 2};
  std::vector<int32_t> tensor_map_expect = {3, 2, 1, 0};
  std::vector<int32_t> tensor_shape_expect = {4, 128, 2, 512};
  ReshapeExpandDeviceArrangementTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_new,
                                             tensor_map_expect, tensor_shape_expect);
}

/*
 *  example2:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [0, 1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [1, 0, 3, 2],
 *    out_tensor_shape = [2, 256, 4, 256]
 */
TEST_F(TestTensorLayout, ReshapeExpandDeviceArrangement2) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {0, 1};
  std::vector<int32_t> tensor_shape = {512, 1024};
  std::vector<int32_t> device_arrangement_new = {4, 2, 2, 2};
  std::vector<int32_t> tensor_map_expect = {1, 0, 3, 2};
  std::vector<int32_t> tensor_shape_expect = {2, 256, 4, 256};
  ReshapeExpandDeviceArrangementTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_new,
                                             tensor_map_expect, tensor_shape_expect);
}

/*
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, -1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [3, 2, -1],
 *    out_tensor_shape = [4, 128, 1024]
 */
TEST_F(TestTensorLayout, ReshapeExpandDeviceArrangement3) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {1, -1};
  std::vector<int32_t> tensor_shape = {512, 1024};
  std::vector<int32_t> device_arrangement_new = {4, 2, 2, 2};
  std::vector<int32_t> tensor_map_expect = {3, 2, -1};
  std::vector<int32_t> tensor_shape_expect = {4, 128, 1024};
  ReshapeExpandDeviceArrangementTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_new,
                                             tensor_map_expect, tensor_shape_expect);
}

/*
 *  example4:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [0, 1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 4]
 *  =>
 *    out_tensor_map = [0, 2, 1],
 *    out_tensor_shape = [512, 4, 256]
 */
TEST_F(TestTensorLayout, ReshapeExpandDeviceArrangement4) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {0, 1};
  std::vector<int32_t> tensor_shape = {512, 1024};
  std::vector<int32_t> device_arrangement_new = {4, 2, 4};
  std::vector<int32_t> tensor_map_expect = {0, 2, 1};
  std::vector<int32_t> tensor_shape_expect = {512, 4, 256};
  ReshapeExpandDeviceArrangementTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_new,
                                             tensor_map_expect, tensor_shape_expect);
}

TEST_F(TestTensorLayout, ReshapeExpandDeviceArrangement5) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {1, -1, 0};
  std::vector<int32_t> tensor_shape = {128, 4, 1024};
  std::vector<int32_t> device_arrangement_new = {8, 4};
  std::vector<int32_t> tensor_map_expect = {1, -1, 0};
  std::vector<int32_t> tensor_shape_expect = {128, 4, 1024};
  ReshapeExpandDeviceArrangementTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_new,
                                             tensor_map_expect, tensor_shape_expect);
}

void ExpandTensorShapeTestFunction(const std::vector<int32_t>& in_device_arrangement_shape,
                                   const std::vector<int32_t>& in_tensor_map_shape,
                                   const std::vector<int32_t>& in_tensor_shape_shape,
                                   const std::vector<int32_t>& out_device_arrangement_shape,
                                   const std::vector<int32_t>& out_tensor_map_shape,
                                   const std::vector<int32_t>& out_tensor_shape_shape) {
  Arrangement device_arrangement;
  Status status = device_arrangement.Init(in_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map tensor_map;
  status = tensor_map.Init(in_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape;
  status = tensor_shape.Init(in_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape_new;
  status = tensor_shape_new.Init(out_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout tensor_layout;
  tensor_layout.Init(device_arrangement, tensor_map, tensor_shape);
  std::shared_ptr<TensorLayout> tensor_layout_new_ptr = tensor_layout.ExpandTensorShape(tensor_shape_new);
  ASSERT_NE(nullptr, tensor_layout_new_ptr);
  Arrangement device_arrangement_new = tensor_layout_new_ptr->device_arrangement();
  ASSERT_EQ(Status::SUCCESS, status);
  ASSERT_EQ(out_device_arrangement_shape, device_arrangement_new.array());
  Map tensor_map_new = tensor_layout_new_ptr->tensor_map();
  ASSERT_EQ(out_tensor_map_shape, tensor_map_new.array());
}

/*
 *  example:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_tensor_shape = [4, 128, 1024]
 *  =>
 *    out_device_arrangement = [4, 2, 4]
 *    out_tensor_map = [2, 1, 0],
 */
TEST_F(TestTensorLayout, ExpandTensorShape1) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {1, 0};
  std::vector<int32_t> tensor_shape = {512, 1024};
  std::vector<int32_t> device_arrangement_expect = {4, 2, 4};
  std::vector<int32_t> tensor_map_expect = {2, 1, 0};
  std::vector<int32_t> tensor_shape_new = {4, 128, 1024};
  ExpandTensorShapeTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_expect,
                                tensor_map_expect, tensor_shape_new);
}

TEST_F(TestTensorLayout, ExpandTensorShape2) {
  std::vector<int32_t> device_arrangement = {8, 4};
  std::vector<int32_t> tensor_map = {1, 0};
  std::vector<int32_t> tensor_shape = {128, 4096};
  std::vector<int32_t> device_arrangement_expect = {8, 4};
  std::vector<int32_t> tensor_map_expect = {1, 0, -1};
  std::vector<int32_t> tensor_shape_new = {128, 4, 1024};
  ExpandTensorShapeTestFunction(device_arrangement, tensor_map, tensor_shape, device_arrangement_expect,
                                tensor_map_expect, tensor_shape_new);
}

TEST_F(TestTensorLayout, GetSliceShape) {
  std::vector<int32_t> in_device_arrangement = {8, 4};
  std::vector<int32_t> in_tensor_map = {1, -1};
  std::vector<int32_t> in_tensor_shape = {512, 1024};
  Arrangement device_arrangement;
  device_arrangement.Init(in_device_arrangement);
  Map tensor_map;
  tensor_map.Init(in_tensor_map);
  Arrangement tensor_shape;
  tensor_shape.Init(in_tensor_shape);
  TensorLayout tensor_layout;
  tensor_layout.Init(device_arrangement, tensor_map, tensor_shape);
  Arrangement new_tensor_shape = tensor_layout.slice_shape();
  Arrangement expected_shape;
  expected_shape.Init({64, 1024});
  ASSERT_EQ(new_tensor_shape, expected_shape);
}

TEST_F(TestTensorLayout, UpdateTensorMap) {
  std::vector<int32_t> in_device_arrangement = {8, 4};
  std::vector<int32_t> in_tensor_map = {1, -1};
  std::vector<int32_t> in_tensor_shape = {512, 1024};
  Arrangement device_arrangement;
  device_arrangement.Init(in_device_arrangement);
  Map tensor_map;
  tensor_map.Init(in_tensor_map);
  Arrangement tensor_shape;
  tensor_shape.Init(in_tensor_shape);
  TensorLayout tensor_layout;
  tensor_layout.Init(device_arrangement, tensor_map, tensor_shape);
  tensor_layout.UpdateTensorMap(1, 0);
  in_tensor_map[1] = 0;
  Shape new_tensor_map = tensor_layout.tensor_map().array();
  ASSERT_EQ(in_tensor_map, new_tensor_map);
}

void RemoveElementEqualToOneInDeviceArrangementTestFunction(const std::vector<int32_t>& in_device_arrangement_shape,
                                                            const std::vector<int32_t>& in_tensor_map_shape,
                                                            const std::vector<int32_t>& in_tensor_shape_shape,
                                                            const std::vector<int32_t>& out_device_arrangement_shape,
                                                            const std::vector<int32_t>& out_tensor_map_shape,
                                                            const std::vector<int32_t>& out_tensor_shape_shape) {
  Arrangement device_arrangement;
  Status status = device_arrangement.Init(in_device_arrangement_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Map tensor_map;
  status = tensor_map.Init(in_tensor_map_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape;
  status = tensor_shape.Init(in_tensor_shape_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  TensorLayout tensor_layout;
  status = tensor_layout.Init(device_arrangement, tensor_map, tensor_shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Arrangement tensor_shape_new = tensor_layout.tensor_shape();
  ASSERT_EQ(out_tensor_shape_shape, tensor_shape_new.array());
  Arrangement device_arrangement_new = tensor_layout.device_arrangement();
  ASSERT_EQ(out_device_arrangement_shape, device_arrangement_new.array());
  Map tensor_map_new = tensor_layout.tensor_map();
  ASSERT_EQ(out_tensor_map_shape, tensor_map_new.array());
}

TEST_F(TestTensorLayout, RemoveElementEqualToOneInDeviceArrangement1) {
  std::vector<int32_t> device_arrangement = {2, 2, 1};
  std::vector<int32_t> tensor_map = {2, 1};
  std::vector<int32_t> tensor_shape = {128, 4096};
  std::vector<int32_t> device_arrangement_expect = {2, 2};
  std::vector<int32_t> tensor_map_expect = {1, 0};
  std::vector<int32_t> tensor_shape_new = {128, 4096};
  RemoveElementEqualToOneInDeviceArrangementTestFunction(
    device_arrangement, tensor_map, tensor_shape, device_arrangement_expect, tensor_map_expect, tensor_shape_new);
}

TEST_F(TestTensorLayout, RemoveElementEqualToOneInDeviceArrangement2) {
  std::vector<int32_t> device_arrangement = {16, 1, 1};
  std::vector<int32_t> tensor_map = {2, 0};
  std::vector<int32_t> tensor_shape = {128, 4096};
  std::vector<int32_t> device_arrangement_expect = {16};
  std::vector<int32_t> tensor_map_expect = {0, -1};
  std::vector<int32_t> tensor_shape_new = {128, 4096};
  RemoveElementEqualToOneInDeviceArrangementTestFunction(
    device_arrangement, tensor_map, tensor_shape, device_arrangement_expect, tensor_map_expect, tensor_shape_new);
}

TEST_F(TestTensorLayout, RemoveElementEqualToOneInDeviceArrangement3) {
  std::vector<int32_t> device_arrangement = {1, 16, 1};
  std::vector<int32_t> tensor_map = {2, 1};
  std::vector<int32_t> tensor_shape = {128, 4096};
  std::vector<int32_t> device_arrangement_expect = {16};
  std::vector<int32_t> tensor_map_expect = {-1, 0};
  std::vector<int32_t> tensor_shape_new = {128, 4096};
  RemoveElementEqualToOneInDeviceArrangementTestFunction(
    device_arrangement, tensor_map, tensor_shape, device_arrangement_expect, tensor_map_expect, tensor_shape_new);
}

TEST_F(TestTensorLayout, RemoveElementEqualToOneInDeviceArrangement4) {
  std::vector<int32_t> device_arrangement = {1, 1, 1};
  std::vector<int32_t> tensor_map = {2, 1};
  std::vector<int32_t> tensor_shape = {128, 4096};
  std::vector<int32_t> device_arrangement_expect = {};
  std::vector<int32_t> tensor_map_expect = {-1, -1};
  std::vector<int32_t> tensor_shape_new = {128, 4096};
  RemoveElementEqualToOneInDeviceArrangementTestFunction(
    device_arrangement, tensor_map, tensor_shape, device_arrangement_expect, tensor_map_expect, tensor_shape_new);
}

}  // namespace parallel
}  // namespace mindspore
