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
#include "parallel/tensor_layout/util_layout_gen_test.h"
#include <cmath>
#include <map>
#include <tuple>
#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
#include "frontend/parallel/tensor_layout/shape_util.h"
#include "common/common_test.h"

using std::pow;

namespace mindspore {
namespace parallel {
std::vector<Shape> combine(const Shape &in, int64_t target) {
  std::vector<Shape> output;
  for (int64_t i = 0; i < pow(2, in.size()); i++) {
    size_t temp = 0;
    size_t count = 0;
    Shape left;
    for (size_t j = 0; j < in.size(); j++) {
      if ((i & (1 << j)) != 0) {
        left.push_back(j);
        count++;
      }
    }
    if (count == target) {
      Shape one_case;
      for (size_t j = 0; j < count; j++) {
        temp = in.size() - 1 - left[j];
        one_case.push_back(in[temp]);
      }
      if (one_case.size() > 0) {
        output.push_back(one_case);
      }
    }
  }
  return output;
}

void GenerateValidShapeBySizeAndDim(int64_t pow_size, int64_t dim, std::vector<Shape> *out) {
  out->clear();
  Shape in;
  for (int64_t i = 1; i < pow_size; i++) {
    in.push_back(i);
  }
  std::vector<Shape> combine_result;
  combine_result = combine(in, dim - 1);
  if (combine_result.size() == 0) {
    int64_t size = exp2(pow_size);
    Shape item = {size};
    out->push_back(item);
  }
  for (size_t i = 0; i < combine_result.size(); i++) {
    Shape item;
    int64_t prev = 0;
    for (int64_t j = combine_result[i].size() - 1; j >= 0; j--) {
      item.push_back(exp2(combine_result[i][j] - prev));
      prev = combine_result[i][j];
    }
    item.push_back(exp2(pow_size - prev));
    out->push_back(item);
  }
  return;
}

void GenerateValidShapeBySize(int64_t pow_size, std::vector<Shape> *out) {
  out->clear();
  for (int64_t dim = 1; dim <= pow_size; dim++) {
    std::vector<Shape> combine_result;
    GenerateValidShapeBySizeAndDim(pow_size, dim, &combine_result);
    for (size_t i = 0; i < combine_result.size(); i++) {
      out->push_back(combine_result[i]);
    }
  }
  return;
}

TensorMap GenerateTensorMap(const int64_t &map_size, const Shape &pos_index, const Shape &pos_value) {
  TensorMap tensor_map(map_size, -1);
  for (size_t i = 0; i < pos_index.size() && i < pos_value.size(); i++) {
    if (pos_index[i] >= map_size) {
      continue;
    }
    tensor_map[pos_index[i]] = pos_value[i];
  }
  return tensor_map;
}

void GenerateValidTensorMap(const DeviceArrangement &device_arrangement, const TensorShape &tensor_shape,
                            std::vector<TensorMap> *tensor_map_list) {
  tensor_map_list->clear();
  int64_t device_size = device_arrangement.size();
  int64_t shape_size = tensor_shape.size();
  Shape pos_ind_combine_in;
  for (int64_t i = 0; i < shape_size; i++) {
    pos_ind_combine_in.push_back(i);
  }
  Shape dev_ind_combine_in;
  for (int64_t i = 0; i < device_size; i++) {
    dev_ind_combine_in.push_back(i);
  }
  TensorMap none_map(tensor_shape.size(), -1);
  tensor_map_list->push_back(none_map);
  for (int64_t pos_num = 1; (pos_num <= shape_size) && (pos_num <= device_size); pos_num++) {
    std::vector<Shape> pos_index;
    pos_index = combine(pos_ind_combine_in, pos_num);
    std::vector<Shape> dev_index;
    dev_index = combine(dev_ind_combine_in, pos_num);
    for (size_t l = 0; l < dev_index.size(); l++) {
      Shape pos_value_combine_in;
      for (int32_t i = dev_index[l].size() - 1; i >= 0; i--) {
        pos_value_combine_in.push_back(dev_index[l][i]);
      }
      std::vector<Shape> pos_value;
      Shape::iterator it = pos_value_combine_in.begin();
      do {
        Shape pos_value_item;
        for (size_t m = 0; m < pos_num; m++) {
          pos_value_item.push_back(pos_value_combine_in[m]);
        }
        pos_value.push_back(pos_value_item);
      } while (next_permutation(it, it + pos_num));
      for (size_t j = 0; j < pos_index.size(); j++) {
        for (size_t k = 0; k < pos_value.size(); k++) {
          TensorMap tensor_map = GenerateTensorMap(shape_size, pos_index[j], pos_value[k]);
          tensor_map_list->push_back(tensor_map);
        }
      }
    }
  }
  return;
}

void GenerateValidLayoutByDeviceSizeAndTensorSize(
  int64_t device_pow_size, int64_t tensor_pow_size, int64_t max_device_dim, int64_t max_shape_dim,
  std::vector<std::tuple<DeviceArrangement, TensorMap, TensorShape>> *layout_list) {
  layout_list->clear();
  std::vector<DeviceArrangement> device_arrangement_list;
  GenerateValidShapeBySize(device_pow_size, &device_arrangement_list);
  std::vector<TensorShape> tensor_shape_list;
  GenerateValidShapeBySize(tensor_pow_size, &tensor_shape_list);
  for (size_t device_idx = 0; device_idx < device_arrangement_list.size(); device_idx++) {
    for (size_t shape_idx = 0; shape_idx < tensor_shape_list.size(); shape_idx++) {
      std::vector<TensorMap> tensor_map_list;
      GenerateValidTensorMap(device_arrangement_list[device_idx], tensor_shape_list[shape_idx], &tensor_map_list);
      for (size_t map_idx = 0; map_idx < tensor_map_list.size(); map_idx++) {
        if (!CheckLayoutValid(device_arrangement_list[device_idx], tensor_map_list[map_idx],
                              tensor_shape_list[shape_idx])) {
          continue;
        }
        layout_list->push_back(
          std::make_tuple(device_arrangement_list[device_idx], tensor_map_list[map_idx], tensor_shape_list[shape_idx]));
      }
    }
  }
  return;
}

bool CheckLayoutValid(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                      const TensorShape &tensor_shape) {
  bool flag = false;
  if ((tensor_map.size() - ComputeNoneNumber(tensor_map)) > device_arrangement.size()) {
    return flag;
  }
  if (!ShapeIsDividedByDevice(device_arrangement, tensor_map, tensor_shape)) {
    return flag;
  }
  return true;
}

size_t ComputeNoneNumber(const TensorMap &tensor_map) {
  size_t num = 0;
  for (size_t i = 0; i < tensor_map.size(); i++) {
    if (tensor_map[i] == -1) {
      num++;
    }
  }
  return num;
}

bool ShapeIsDividedByDevice(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                            const TensorShape &tensor_shape) {
  bool flag = false;
  for (uint32_t i = 0; i < tensor_map.size() && i < tensor_shape.size(); i++) {
    if (tensor_map[i] == -1) {
      continue;
    }
    int64_t dim = device_arrangement[device_arrangement.size() - 1 - tensor_map[i]];
    if (tensor_shape[i] % dim != 0) {
      return flag;
    }
  }
  return true;
}

bool IsExpended(const Shape &in1, const Shape &in2) {
  int64_t size = 1;
  uint32_t ind = 0;
  for (uint32_t i = 0; i < in1.size(); i++) {
    size *= in1[i];
    if (ind >= in2.size()) {
      return false;
    }
    if (size > in2[ind]) {
      return false;
    } else if (size < in2[ind]) {
      continue;
    } else {
      ind++;
      size = 1;
    }
  }
  if (ind != in2.size()) {
    return false;
  }
  return true;
}

void ComputeAccumDeviceTOAccumShapeMap(const DeviceArrangement &device_arrangement, const TensorMap &tensor_map,
                                       const TensorShape &tensor_shape,
                                       std::map<int64_t, int64_t> *accum_device_to_accum_shape_map) {
  accum_device_to_accum_shape_map->clear();
  std::vector<int64_t> shape_accum_reverse;
  Status status = ShapeToAccumulateProductReverse(tensor_shape, &shape_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> device_accum_reverse;
  status = ShapeToAccumulateProductReverse(device_arrangement, &device_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  for (int32_t i = 0; i < device_accum_reverse.size(); i++) {
    auto iter = std::find(tensor_map.begin(), tensor_map.end(), device_accum_reverse.size() - 1 - i);
    if (iter == tensor_map.end()) {
      accum_device_to_accum_shape_map->insert(std::make_pair(device_accum_reverse[i], -1));
    } else {
      accum_device_to_accum_shape_map->insert(
        std::make_pair(device_accum_reverse[i], shape_accum_reverse[std::distance(tensor_map.begin(), iter)]));
    }
  }
  return;
}

void IsLinearValue(int64_t small, int64_t big, int64_t small_value, int64_t big_value, int64_t middle,
                   int64_t middle_value) {
  ASSERT_NE(big, small);
  int64_t value = (middle - small) * (big_value - small_value) / (big - small) + small_value;
  ASSERT_EQ(middle_value, value);
}

void LayoutTransferValidLayoutChangeCheck(const DeviceArrangement &in_device_arrangement,
                                          const TensorMap &in_tensor_map, const TensorShape &in_tensor_shape,
                                          const DeviceArrangement &out_device_arrangement,
                                          const TensorMap &out_tensor_map, const TensorShape &out_tensor_shape) {
  bool is_expended = IsExpended(out_device_arrangement, in_device_arrangement);
  ASSERT_EQ(true, is_expended);
  is_expended = IsExpended(out_tensor_shape, in_tensor_shape);
  ASSERT_EQ(true, is_expended);
  std::map<int64_t, int64_t> out_accum_device_to_accum_shape_map;
  ComputeAccumDeviceTOAccumShapeMap(out_device_arrangement, out_tensor_map, out_tensor_shape,
                                    &out_accum_device_to_accum_shape_map);
  std::map<int64_t, int64_t> in_accum_device_to_accum_shape_map;
  ComputeAccumDeviceTOAccumShapeMap(in_device_arrangement, in_tensor_map, in_tensor_shape,
                                    &in_accum_device_to_accum_shape_map);
  std::map<int64_t, int64_t>::iterator in_iter = in_accum_device_to_accum_shape_map.begin();
  while (in_iter != in_accum_device_to_accum_shape_map.end()) {
    if (in_iter->second != out_accum_device_to_accum_shape_map[in_iter->first]) {
      continue;
    }
    in_iter++;
  }
  std::map<int64_t, int64_t>::iterator out_iter = out_accum_device_to_accum_shape_map.begin();
  while (out_iter != out_accum_device_to_accum_shape_map.end()) {
    if (out_accum_device_to_accum_shape_map.find(out_iter->first) == out_accum_device_to_accum_shape_map.end()) {
      in_iter = in_accum_device_to_accum_shape_map.begin();
      int64_t small = 1;
      int64_t big = 1;
      while (in_iter != in_accum_device_to_accum_shape_map.end()) {
        if (in_iter->first < out_iter->first) {
          small = in_iter->second;
        } else if (in_iter->first > out_iter->first) {
          big = in_iter->second;
          break;
        } else {
          ASSERT_EQ(true, false);
        }
        in_iter++;
      }
      if (small == 1) {
        ASSERT_EQ(true, false);
      }
      if (big == 1) {
        ASSERT_EQ(true, false);
      }
      int64_t small_value = in_accum_device_to_accum_shape_map[small];
      int64_t big_value = in_accum_device_to_accum_shape_map[big];
      IsLinearValue(small, big, small_value, big_value, out_iter->first, out_iter->second);
    }
    out_iter++;
  }
}

void ValidLayoutChangeCheck(const DeviceArrangement &in_device_arrangement, const TensorMap &in_tensor_map,
                            const TensorShape &in_tensor_shape, const DeviceArrangement &out_device_arrangement,
                            const TensorMap &out_tensor_map, const TensorShape &out_tensor_shape) {
  LayoutTransferValidLayoutChangeCheck(in_device_arrangement, in_tensor_map, in_tensor_shape, out_device_arrangement,
                                       out_tensor_map, out_tensor_shape);
}

}  // namespace parallel
}  // namespace mindspore
