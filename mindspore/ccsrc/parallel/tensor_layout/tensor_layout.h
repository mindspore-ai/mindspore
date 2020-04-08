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

#ifndef MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_
#define MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "parallel/device_manager.h"
#include "parallel/status.h"
#include "parallel/tensor_layout/arrangement.h"
#include "parallel/tensor_layout/map.h"
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
  Status Init(const Arrangement& device_arrangement, const Map& tensor_map, const Arrangement& tensor_shape);
  Status InitFromVector(const std::vector<int32_t>& device_arrangement, const std::vector<int32_t>& tensor_map,
                        const std::vector<int32_t>& tensor_shape);

  Arrangement device_arrangement() const { return device_arrangement_; }

  Map tensor_map() const { return tensor_map_; }

  Arrangement tensor_shape() const { return tensor_shape_; }

  Map origin_tensor_map() const { return tensor_map_origin_; }

  std::shared_ptr<TensorLayout> ExpandTensorShape(const Arrangement& expanded_shape) const;

  std::shared_ptr<TensorLayout> ExpandDeviceArrangement(const Arrangement& expanded_arrangement) const;

  bool IsSameTensorShape(const TensorLayout& tensor_layout) const {
    return (tensor_shape_ == tensor_layout.tensor_shape());
  }

  bool IsSameDeviceArrangement(const TensorLayout& tensor_layout) const {
    return (device_arrangement_ == tensor_layout.device_arrangement());
  }

  bool IsSameTensorMap(const TensorLayout& tensor_layout) const { return (tensor_map_ == tensor_layout.tensor_map()); }

  bool operator==(const TensorLayout& t1) const;

  bool TensorShapeCanBeExpanded(const Arrangement& expanded_shape) const;

  std::shared_ptr<Arrangement> ComputeExpandedTensorShape(const Arrangement& expand_shape) const;

  Arrangement slice_shape() const;

  Status UpdateTensorMap(uint32_t index, int32_t value);

  TensorLayout SqueezeShape() const;

 private:
  std::shared_ptr<TensorLayout> ExpandTensorShapeWithoutExtendDeviceArrangement(
    const Arrangement& expanded_shape) const;
  std::shared_ptr<Arrangement> ComputeArrangementByExpandedShape(const Arrangement& tensor_shape) const;
  bool IsValidTensorLayout() const;
  void RemoveElementEqualToOneInDeviceArrangement();
  int32_t GetSliceDeviceDimensionByTensorDimensionIndex(uint32_t idx) const;
  int32_t GetSliceNumByTensorDimensionIndex(uint32_t idx) const;
  bool TensorShapeDimensionIsDividedBySplitDeviceDimension() const;
  int32_t GetTensorDimensionIndexByDeviceDimensionIndex(int32_t idx) const;

  Arrangement device_arrangement_origin_;
  Map tensor_map_origin_;
  Arrangement tensor_shape_origin_;
  Arrangement device_arrangement_;
  Map tensor_map_;
  Arrangement tensor_shape_;
};

}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_
