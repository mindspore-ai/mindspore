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
#ifndef TESTS_UT_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_
#define TESTS_UT_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_

#include <map>
#include <tuple>
#include <vector>

#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

std::vector<Shape> combine(const Shape &in, int64_t target);

void GenerateValidShapeBySizeAndDim(int64_t pow_size, int64_t dim, std::vector<Shape> *out);

void GenerateValidShapeBySize(int64_t pow_size, std::vector<Shape> *out);

TensorMap GenerateTensorMap(const int64_t &map_size, const Shape &pos_index, const Shape &pos_value);

void GenerateValidTensorMap(const DeviceArrangement &device_arrangement, const TensorMap &tensor_shape,
                            std::vector<TensorMap> *tensor_map_list);

void GenerateValidLayoutByDeviceSizeAndTensorSize(
  int64_t device_pow_size, int64_t tensor_pow_size, int64_t max_device_dim, int64_t max_shape_dim,
  std::vector<std::tuple<DeviceArrangement, TensorMap, TensorShape>> *layout_list);

size_t ComputeNoneNumber(const TensorMap &tensor_map);

bool ShapeIsDividedByDevice(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                            const TensorShape &tensor_shape);

bool CheckLayoutValid(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                      const TensorShape &tensor_shape);

void ComputeAccumDeviceTOAccumShapeMap(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                                       const TensorShape &tensor_shape,
                                       std::map<int64_t, int64_t> *accum_device_to_accum_shape_map);

void LayoutTransferValidLayoutChangeCheck(const DeviceArrangement &in_device_arrangement,
                                          const TensorMap &in_tensor_map, const TensorShape &in_tensor_shape,
                                          const DeviceArrangement &out_device_arrangement,
                                          const TensorMap &out_tensor_map, const TensorShape &out_tensor_shape);

void ValidLayoutChangeCheck(const DeviceArrangement &in_device_arrangement, const TensorMap &in_tensor_map,
                            const TensorShape &in_tensor_shape, const DeviceArrangement &out_device_arrangement,
                            const TensorMap &out_tensor_map, const TensorShape &out_tensor_shape);

}  // namespace parallel
}  // namespace mindspore
#endif  // TESTS_UT_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_
