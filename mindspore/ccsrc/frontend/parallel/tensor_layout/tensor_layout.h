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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/arrangement.h"
#include "frontend/parallel/tensor_layout/map.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace parallel {
class TensorLayout {
 public:
  TensorLayout() = default;
  ~TensorLayout() = default;
  std::string ToString() const;
  std::string StandardToString() const;
  std::string OriginToString() const;
  Status Init(const Arrangement &device_arrangement, const Map &tensor_map, const Arrangement &tensor_shape);
  Status InitFromVector(const Shape &device_arrangement, const Shape &tensor_map, const Shape &tensor_shape);

  bool skip_redistribution() const { return skip_redistribution_; }

  void set_skip_redistribution(bool flag) { skip_redistribution_ = flag; }

  int32_t get_field_size() const { return field_size_; }

  void set_field_size(int32_t field_size) { field_size_ = field_size; }

  bool uniform_split() const { return uniform_split_; }

  void set_uniform_split(bool flag) { uniform_split_ = flag; }

  Arrangement device_arrangement() const { return device_arrangement_; }

  Map tensor_map() const { return tensor_map_; }

  Arrangement tensor_shape() const { return tensor_shape_; }

  Map origin_tensor_map() const { return tensor_map_origin_; }

  std::shared_ptr<TensorLayout> ExpandTensorShape(const Arrangement &expanded_shape) const;

  std::shared_ptr<TensorLayout> ExpandDeviceArrangement(const Arrangement &expanded_arrangement) const;

  bool IsSameTensorShape(const TensorLayout &tensor_layout) const {
    return (tensor_shape_ == tensor_layout.tensor_shape());
  }

  bool IsSameDeviceArrangement(const TensorLayout &tensor_layout) const {
    return (device_arrangement_ == tensor_layout.device_arrangement());
  }

  bool IsSameTensorMap(const TensorLayout &tensor_layout) const { return (tensor_map_ == tensor_layout.tensor_map()); }

  bool operator==(const TensorLayout &t1) const;

  bool TensorShapeCanBeExpanded(const Arrangement &expanded_shape) const;

  std::shared_ptr<Arrangement> ComputeExpandedTensorShape(const Arrangement &expand_shape) const;

  Arrangement slice_shape() const;

  Status UpdateTensorMap(size_t index, int64_t value);

  TensorLayout SqueezeShape() const;

  // Key for user data.
  constexpr static char key[] = "TLayout";

 private:
  std::shared_ptr<TensorLayout> ExpandTensorShapeWithoutExtendDeviceArrangement(
    const Arrangement &expanded_shape) const;
  std::shared_ptr<Arrangement> ComputeArrangementByExpandedShape(const Arrangement &tensor_shape) const;
  bool IsValidTensorLayout() const;
  void RemoveElementEqualToOneInDeviceArrangement();
  int32_t GetSliceDeviceDimensionByTensorDimensionIndex(uint32_t idx) const;
  int32_t GetSliceNumByTensorDimensionIndex(uint32_t idx) const;
  bool TensorShapeDimensionIsDividedBySplitDeviceDimension() const;
  int32_t GetTensorDimensionIndexByDeviceDimensionIndex(int64_t idx) const;

  Arrangement device_arrangement_origin_;
  Map tensor_map_origin_;
  Arrangement tensor_shape_origin_;
  Arrangement device_arrangement_;
  Map tensor_map_;
  Arrangement tensor_shape_;
  bool skip_redistribution_ = false;
  int32_t field_size_ = 0;
  bool uniform_split_ = true;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_
