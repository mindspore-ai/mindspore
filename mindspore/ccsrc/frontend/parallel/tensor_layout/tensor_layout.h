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
#include <utility>
#include <vector>
#include <functional>
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/arrangement.h"
#include "frontend/parallel/tensor_layout/map.h"
#include "include/common/utils/convert_utils.h"

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
  Status InitFromExtendVector(const Shape &device_arrangement, const std::vector<Shape> &tensor_map,
                              const Shape &tensor_shape);
  bool skip_redistribution() const { return skip_redistribution_; }

  void set_skip_redistribution(bool flag) { skip_redistribution_ = flag; }

  bool layout_transfer() const { return layout_transfer_; }

  void set_layout_transfer(bool flag) { layout_transfer_ = flag; }

  int64_t get_field_size() const { return field_size_; }

  void set_field_size(int64_t field_size) { field_size_ = field_size; }

  bool uniform_split() const { return uniform_split_; }

  void set_uniform_split(bool flag) { uniform_split_ = flag; }

  Arrangement device_arrangement() const { return device_arrangement_; }

  Map tensor_map() const { return tensor_map_; }

  Arrangement tensor_shape() const { return tensor_shape_; }

  Arrangement tensor_shape_origin() const { return tensor_shape_origin_; }

  Arrangement device_arrangement_origin() const { return device_arrangement_origin_; }

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

  bool operator!=(const TensorLayout &t1) const;

  bool IsSameWithoutSplit(const TensorLayout &t1) const;

  bool TensorShapeCanBeExpanded(const Arrangement &expand_shape) const;

  std::shared_ptr<Arrangement> ComputeExpandedTensorShape(const Arrangement &expand_shape) const;

  Arrangement slice_shape() const;

  Arrangement base_slice_shape() const;

  Shape shard_strategy() const;

  Status UpdateTensorMap(size_t index, int64_t value);

  TensorLayout SqueezeShape() const;

  TensorLayout TransferRepeatLayout() const;

  Status GenerateOptShardSliceShape();

  Shape opt_shard_slice_shape() { return opt_shard_slice_shape_; }

  void set_opt_shard_group(std::string name) { opt_shard_group_ = std::move(name); }

  std::string opt_shard_group() const { return opt_shard_group_; }

  void set_opt_shard_mirror_group(std::string name) { opt_shard_mirror_group_ = std::move(name); }

  std::string opt_shard_mirror_group() { return opt_shard_mirror_group_; }

  void set_opt_weight_shard_step(int32_t step) { opt_weight_shard_step_ = step; }

  int32_t opt_weight_shard_step() const { return opt_weight_shard_step_; }

  void set_opt_weight_shard_size(int32_t size) { opt_weight_shard_size_ = size; }

  int32_t opt_weight_shard_size() const { return opt_weight_shard_size_; }

  void set_is_shared_param(bool is_shared_param) { is_shared_param_ = is_shared_param; }

  bool is_shared_param() const { return is_shared_param_; }

  void set_tensor_shape_before(const Shape &tensor_shape_before) { tensor_shape_before_.Init(tensor_shape_before); }

  RankList InferRepeatedGroup();

  Arrangement tensor_shape_before() { return tensor_shape_before_; }

  std::vector<Shape> tensor_map_before() { return tensor_map_before_; }
  // Key for user data.
  constexpr static char key[] = "TLayout";

 private:
  std::shared_ptr<TensorLayout> ExpandTensorShapeWithoutExtendDeviceArrangement(
    const Arrangement &expanded_shape) const;
  std::shared_ptr<Arrangement> ComputeArrangementByExpandedShape(const Arrangement &tensor_shape) const;
  bool IsValidTensorLayout() const;
  void RemoveElementEqualToOneInDeviceArrangement();
  int64_t GetSliceDeviceDimensionByTensorDimensionIndex(uint64_t idx) const;
  int64_t GetSliceNumByTensorDimensionIndex(uint64_t idx) const;
  bool TensorShapeDimensionIsDividedBySplitDeviceDimension() const;
  int64_t GetTensorDimensionIndexByDeviceDimensionIndex(int64_t idx) const;

  Arrangement device_arrangement_origin_;
  Arrangement tensor_shape_origin_;
  Arrangement device_arrangement_;
  Arrangement tensor_shape_;
  Arrangement tensor_shape_before_;
  Map tensor_map_;
  Map tensor_map_origin_;
  std::vector<Shape> tensor_map_before_;
  bool skip_redistribution_ = false;
  bool uniform_split_ = true;
  bool layout_transfer_ = false;
  int64_t field_size_ = 0;
  Shape opt_shard_slice_shape_;
  std::string opt_shard_group_ = "";         // for allgather
  std::string opt_shard_mirror_group_ = "";  // for mirror ops
  int32_t opt_weight_shard_step_ = 0;
  int32_t opt_weight_shard_size_ = 0;
  bool is_shared_param_ = false;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_LAYOUT_H_
