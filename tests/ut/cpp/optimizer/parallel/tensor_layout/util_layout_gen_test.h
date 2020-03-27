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
#ifndef TESTS_UT_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_
#define TESTS_UT_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_

#include <map>
#include <tuple>
#include <vector>

#include "optimizer/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {

std::vector<std::vector<int32_t>> combine(const std::vector<int32_t>& in, int32_t target);

void GenerateValidShapeBySizeAndDim(const int32_t& pow_size, const int32_t& dim,
                                    std::vector<std::vector<int32_t>>* out);

void GenerateValidShapeBySize(const int32_t& pow_size, std::vector<std::vector<int32_t>>* out);

std::vector<int32_t> GenerateTensorMap(const uint32_t& map_size, const std::vector<int32_t>& pos_index,
                                       const std::vector<int32_t>& pos_value);

void GenerateValidTensorMap(const std::vector<int32_t>& device_arrangement, const std::vector<int32_t>& tensor_shape,
                            std::vector<std::vector<int32_t>>* tensor_map_list);

void GenerateValidLayoutByDeviceSizeAndTensorSize(
  const int32_t& device_pow_size, const int32_t& tensor_pow_size, const int32_t& max_device_dim,
  const int32_t& max_shape_dim,
  std::vector<std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>>>* layout_list);

uint32_t ComputeNoneNumber(const std::vector<int32_t>& tensor_map);

bool ShapeIsDividedByDevice(const std::vector<int32_t>& device_arrangement, const std::vector<int32_t>& tensor_map,
                            const std::vector<int32_t>& tensor_shape);

bool CheckLayoutValid(const std::vector<int32_t>& device_arrangement, const std::vector<int32_t>& tensor_map,
                      const std::vector<int32_t>& tensor_shape);

void ComputeAccumDeviceTOAccumShapeMap(const std::vector<int32_t>& device_arrangement,
                                       const std::vector<int32_t>& tensor_map, const std::vector<int32_t>& tensor_shape,
                                       std::map<int32_t, int32_t>* accum_device_to_accum_shape_map);

void LayoutTransferValidLayoutChangeCheck(const std::vector<int32_t>& in_device_arrangement,
                                          const std::vector<int32_t>& in_tensor_map,
                                          const std::vector<int32_t>& in_tensor_shape,
                                          const std::vector<int32_t>& out_device_arrangement,
                                          const std::vector<int32_t>& out_tensor_map,
                                          const std::vector<int32_t>& out_tensor_shape);

void ValidLayoutChangeCheck(const std::vector<int32_t>& in_device_arrangement,
                            const std::vector<int32_t>& in_tensor_map, const std::vector<int32_t>& in_tensor_shape,
                            const std::vector<int32_t>& out_device_arrangement,
                            const std::vector<int32_t>& out_tensor_map, const std::vector<int32_t>& out_tensor_shape);

}  // namespace parallel
}  // namespace mindspore
#endif  // TESTS_UT_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_UT_UTIL_LAYOUT_GEN_H_
