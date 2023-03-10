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

#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include <iostream>
#include <utility>
#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/status.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/shape_util.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
std::string TensorLayout::ToString() const { return StandardToString() + OriginToString(); }

std::string TensorLayout::StandardToString() const {
  std::ostringstream buffer;
  buffer << std::endl << std::string("device arrangement = " + device_arrangement_.ToString());
  buffer << std::endl << std::string("tensor map = " + tensor_map_.ToString());
  buffer << std::endl << std::string("tensor shape = " + tensor_shape_.ToString());
  return buffer.str();
}

std::string TensorLayout::OriginToString() const {
  std::ostringstream buffer;
  buffer << std::endl << std::string("device arrangement origin = " + device_arrangement_origin_.ToString());
  buffer << std::endl << std::string("tensor map origin = " + tensor_map_origin_.ToString());
  buffer << std::endl << std::string("tensor shape origin = " + tensor_shape_origin_.ToString());
  return buffer.str();
}

Status TensorLayout::Init(const Arrangement &device_arrangement, const Map &tensor_map,
                          const Arrangement &tensor_shape) {
  device_arrangement_origin_ = device_arrangement;
  tensor_map_origin_ = tensor_map;
  tensor_shape_origin_ = tensor_shape;
  device_arrangement_ = device_arrangement;
  tensor_map_ = tensor_map;
  tensor_shape_ = tensor_shape;
  if (IsValidTensorLayout()) {
    MS_LOG(DEBUG) << "valid origin tensor layout " << this->OriginToString();
    RemoveElementEqualToOneInDeviceArrangement();
    MS_LOG(DEBUG) << "standard tensor layout " << this->StandardToString();
    return Status::SUCCESS;
  } else {
    if (layout_transfer_) {
      MS_LOG(DEBUG) << "invalid origin tensor layout " << this->OriginToString();
    } else {
      MS_LOG(ERROR) << "invalid origin tensor layout " << this->OriginToString();
    }
    return Status::FAILED;
  }
}

Status TensorLayout::InitFromVector(const Shape &device_arrangement, const Shape &tensor_map,
                                    const Shape &tensor_shape) {
  if (device_arrangement_origin_.Init(device_arrangement) != SUCCESS) {
    return FAILED;
  }
  if (tensor_map_origin_.Init(tensor_map) != SUCCESS) {
    return FAILED;
  }
  if (tensor_shape_origin_.Init(tensor_shape) != SUCCESS) {
    return FAILED;
  }
  if (Init(device_arrangement_origin_, tensor_map_origin_, tensor_shape_origin_) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

bool TensorLayout::IsValidTensorLayout() const {
  if (tensor_map_origin_.GetMaxItem() >= static_cast<int64_t>(device_arrangement_origin_.GetDimSize())) {
    MS_LOG(ERROR) << "the max element in tensor_map_origin_ must be smaller than device_arrangement_origin_ size!";
    return false;
  }
  if (tensor_map_origin_.GetDimSize() != tensor_shape_origin_.GetDimSize()) {
    MS_LOG(ERROR) << "tensor_map_origin_ size must be equal to tensor_shape_origin_ size!";
    return false;
  }
  if (!TensorShapeDimensionIsDividedBySplitDeviceDimension()) {
    if (layout_transfer_) {
      MS_LOG(DEBUG) << "TensorShapeDimensionIsDividedBySplitDeviceDimension failed!";
    } else {
      MS_LOG(ERROR) << "TensorShapeDimensionIsDividedBySplitDeviceDimension failed!";
    }
    return false;
  }
  return true;
}

bool TensorLayout::TensorShapeDimensionIsDividedBySplitDeviceDimension() const {
  for (uint64_t i = 0; i < tensor_map_.GetDimSize(); i++) {
    if (tensor_map_.GetDimByIdx(i) != -1) {
      int64_t divisor = GetSliceNumByTensorDimensionIndex(i);
      if (divisor == 0) {
        MS_LOG(ERROR) << "GetSliceNumByTensorDimensionIndex is 0";
        return false;
      }
      if (tensor_shape_.GetDimByIdx(i) % divisor != 0) {
        return false;
      }
    }
  }
  return true;
}

void TensorLayout::RemoveElementEqualToOneInDeviceArrangement() {
  Shape device_arrangement_shape;
  Shape tensor_map_shape = tensor_map_origin_.array();
  size_t dev_num = device_arrangement_origin_.GetDimSize();
  size_t dev_num_left = device_arrangement_origin_.GetDimSize();
  for (size_t i = 0; i < dev_num; i++) {
    if (device_arrangement_origin_.GetDimByIdx(i) == 1) {
      int64_t idx = GetTensorDimensionIndexByDeviceDimensionIndex(static_cast<int64_t>(dev_num - 1 - i));
      if (idx != -1) {
        tensor_map_shape[static_cast<uint64_t>(idx)] = -1;
      }
      for (auto &value : tensor_map_shape) {
        if (value >= SizeToLong(dev_num_left) - 1 - static_cast<int64_t>(i)) {
          value--;
        }
      }
      continue;
    }
    device_arrangement_shape.push_back(device_arrangement_origin_.GetDimByIdx(i));
  }
  (void)device_arrangement_.Init(device_arrangement_shape);
  (void)tensor_map_.Init(tensor_map_shape);
  tensor_shape_ = tensor_shape_origin_;
}

// if idx is not in tensor_map, return -1
int64_t TensorLayout::GetTensorDimensionIndexByDeviceDimensionIndex(int64_t idx) const {
  return tensor_map_.GetIndexByValue(idx);
}

// tensor_map_.GetDimByIdx(idx) should not be -1
int64_t TensorLayout::GetSliceDeviceDimensionByTensorDimensionIndex(uint64_t idx) const {
  return static_cast<int64_t>(device_arrangement_.GetDimSize()) - 1 - tensor_map_.GetDimByIdx(idx);
}

// tensor_map_.GetDimByIdx(idx) should not be -1
int64_t TensorLayout::GetSliceNumByTensorDimensionIndex(uint64_t idx) const {
  return device_arrangement_.GetDimByIdx(static_cast<uint64_t>(GetSliceDeviceDimensionByTensorDimensionIndex(idx)));
}

std::shared_ptr<TensorLayout> TensorLayout::ExpandTensorShape(const Arrangement &expanded_shape) const {
  std::shared_ptr<Arrangement> expanded_arrangement_ptr = ComputeArrangementByExpandedShape(expanded_shape);
  if (expanded_arrangement_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<TensorLayout> temp_tensor_layout_ptr = ExpandDeviceArrangement(*expanded_arrangement_ptr);
  if (temp_tensor_layout_ptr == nullptr) {
    return nullptr;
  }
  return temp_tensor_layout_ptr->ExpandTensorShapeWithoutExtendDeviceArrangement(expanded_shape);
}

/*
 *  example1:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_tensor_shape = [128, 4, 2, 512],
 *  =>
 *    out_device_arrangement = [8, 2, 2]
 */
std::shared_ptr<Arrangement> TensorLayout::ComputeArrangementByExpandedShape(const Arrangement &tensor_shape) const {
  std::shared_ptr<std::vector<Arrangement>> expand_list_ptr = tensor_shape_.GetExpandShapeList(tensor_shape);
  if (expand_list_ptr == nullptr) {
    return nullptr;
  }
  std::vector<Arrangement> re_map_expand_list;
  Arrangement empty_arrangement;
  for (int64_t i = static_cast<int64_t>(device_arrangement_.GetDimSize()) - 1; i >= 0; i--) {
    if (tensor_map_.GetIndexByValue(i) < 0) {
      re_map_expand_list.push_back(empty_arrangement);
    } else {
      re_map_expand_list.push_back((*expand_list_ptr)[LongToUlong(tensor_map_.GetIndexByValue(i))]);
    }
  }
  std::shared_ptr<Arrangement> new_arrangement_ptr =
    device_arrangement_.GetExpandedShapeByExpandListRemoveLeft(re_map_expand_list);
  return new_arrangement_ptr;
}

/*
 *  example1:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_tensor_shape = [8, 64, 4, 256]
 *  =>
 *    out_device_arrangement = [8, 4],
 *    out_tensor_map = [1, -1, 0, -1],
 */
std::shared_ptr<TensorLayout> TensorLayout::ExpandTensorShapeWithoutExtendDeviceArrangement(
  const Arrangement &expanded_shape) const {
  std::shared_ptr<std::pair<std::vector<Arrangement>, Arrangement>> expand_list_pair_ptr =
    tensor_shape_.GetExpandShapeListPair(expanded_shape);
  if (expand_list_pair_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<Map> tensor_map_new_ptr = tensor_map_.ExpandMapByNone(expand_list_pair_ptr->second);
  if (tensor_map_new_ptr == nullptr) {
    return nullptr;
  }
  TensorLayout tensor_layout_new;
  tensor_layout_new.set_layout_transfer(true);
  Status status = tensor_layout_new.Init(device_arrangement_, *tensor_map_new_ptr, expanded_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<TensorLayout>(tensor_layout_new);
}

/*
 *  example1:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, 0],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [3, 2, 1, 0],
 *    out_tensor_shape = [4, 128, 2, 512]
 *
 *  example2:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [0, 1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [1, 0, 3, 2],
 *    out_tensor_shape = [2, 256, 4, 256]
 *
 *  example3:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [1, -1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 2, 2]
 *  =>
 *    out_tensor_map = [3, 2, -1],
 *    out_tensor_shape = [4, 128, 1024]
 *
 *  example4:
 *    in_device_arrangement = [8, 4],
 *    in_tensor_map = [0, 1],
 *    in_tensor_shape = [512, 1024],
 *    out_device_arrangement = [4, 2, 4]
 *  =>
 *    out_tensor_map = [0, 2, 1],
 *    out_tensor_shape = [512, 4, 256]
 */
std::shared_ptr<TensorLayout> TensorLayout::ExpandDeviceArrangement(const Arrangement &expanded_arrangement) const {
  std::shared_ptr<std::pair<std::vector<Arrangement>, Arrangement>> expand_list_pair_ptr =
    device_arrangement_.GetExpandShapeListPair(expanded_arrangement);
  if (expand_list_pair_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<Map> tensor_map_new_ptr = tensor_map_.ExpandMapByDecreaseNumber(expand_list_pair_ptr->second);
  if (tensor_map_new_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<std::vector<Arrangement>> re_map_shape_list_ptr =
    tensor_map_.ReMapVector(expand_list_pair_ptr->first);
  if (re_map_shape_list_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<Arrangement> tensor_shape_new_ptr =
    tensor_shape_.GetExpandedShapeByExpandListReserveLeft(*re_map_shape_list_ptr);
  if (tensor_shape_new_ptr == nullptr) {
    return nullptr;
  }
  TensorLayout tensor_layout_new;
  Status status = tensor_layout_new.Init(expanded_arrangement, *tensor_map_new_ptr, *tensor_shape_new_ptr);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<TensorLayout>(tensor_layout_new);
}

bool TensorLayout::TensorShapeCanBeExpanded(const Arrangement &expand_shape) const {
  Shape in_expand_shape_shape;
  Status status = ExpandShape(tensor_shape_.array(), expand_shape.array(), &in_expand_shape_shape);
  if (status != Status::SUCCESS) {
    return false;
  }
  return (in_expand_shape_shape == tensor_shape_.array());
}

std::shared_ptr<Arrangement> TensorLayout::ComputeExpandedTensorShape(const Arrangement &expand_shape) const {
  Shape in_expand_shape_shape;
  Status status = ExpandShape(tensor_shape_.array(), expand_shape.array(), &in_expand_shape_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  Arrangement expanded_shape;
  status = expanded_shape.Init(in_expand_shape_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<Arrangement>(expanded_shape);
}

Arrangement TensorLayout::slice_shape() const {
  Shape shape;
  for (size_t index = 0; index < tensor_map_.GetDimSize(); index++) {
    int64_t dim = tensor_map_.GetDimByIdx(index);
    int64_t num = tensor_shape_.GetDimByIdx(index);
    if (dim == -1) {
      shape.push_back(num);
    } else {
      int64_t divisor = device_arrangement_.GetDimByReverseIdx(LongToUlong(dim));
      shape.push_back(num / divisor);
    }
  }
  Arrangement new_tensor_shape;
  if (new_tensor_shape.Init(shape) == Status::FAILED) {
    ValuePtr ptr = MakeValue(shape);
    MS_LOG(EXCEPTION) << "Can't get slice shape when initialize a new shape " << ptr->ToString();
  } else {
    return new_tensor_shape;
  }
}

Status TensorLayout::UpdateTensorMap(size_t index, int64_t value) {
  if (index >= tensor_map_.GetDimSize()) {
    MS_LOG(ERROR) << "Index is out of the size of the tensor map!";
    return Status::FAILED;
  }
  auto shape = tensor_map_.array();
  shape[index] = value;
  if (tensor_map_.Init(shape) == Status::FAILED) {
    MS_LOG(ERROR) << "Update tensor map failed!";
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

RankList GetDevListByTensorMapValue(DeviceMatrix dev_matrix, int64_t tensor_map_value, size_t dev_matrix_size) {
  RankList rank_list;
  if (tensor_map_value >= SizeToLong(dev_matrix_size) || tensor_map_value < MAP_NONE) {
    MS_LOG(ERROR) << "The size of dev_matrix is " << dev_matrix_size << ", but the tensor map value is "
                  << tensor_map_value;
    return rank_list;
  }

  if (tensor_map_value == MAP_NONE) {
    rank_list.push_back(g_device_manager->global_rank());
    return rank_list;
  }

  uint64_t dim = dev_matrix_size - LongToSize(tensor_map_value) - 1;
  if (dev_matrix.GetDevicesAlongDim(dim, &rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "Get devices along dim failed";
  }

  return rank_list;
}

bool TensorLayout::operator==(const TensorLayout &t1) const {
  if (!IsSameTensorShape(t1)) {
    return false;
  }

  if (IsSameDeviceArrangement(t1) && IsSameTensorMap(t1)) {
    return true;
  }

  Shape t1_tensor_map = t1.tensor_map().array();
  Shape this_tensor_map = this->tensor_map().array();
  if (t1_tensor_map.size() != this_tensor_map.size()) {
    return false;
  }

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix t1_dev_matrix(rank, g_device_manager->GetDeviceListInThisStage(), t1.device_arrangement().array());
  DeviceMatrix this_dev_matrix(rank, g_device_manager->GetDeviceListInThisStage(), this->device_arrangement().array());
  size_t t1_dev_mat_size = t1.device_arrangement().array().size();
  size_t this_dev_mat_size = this->device_arrangement().array().size();

  for (size_t i = 0; i < t1_tensor_map.size(); ++i) {
    if (t1_tensor_map[i] == MAP_NONE && this_tensor_map[i] == MAP_NONE) {
      continue;
    }

    RankList t1_dev_list_by_dim = GetDevListByTensorMapValue(t1_dev_matrix, t1_tensor_map[i], t1_dev_mat_size);
    RankList this_dev_list_by_dim = GetDevListByTensorMapValue(this_dev_matrix, this_tensor_map[i], this_dev_mat_size);
    if (t1_dev_list_by_dim.empty() || this_dev_list_by_dim.empty()) {
      MS_LOG(EXCEPTION) << "Can not get device list by tensor map value, these layouts are " << this->ToString()
                        << std::endl
                        << " and " << t1.ToString();
    }

    if (t1_dev_list_by_dim != this_dev_list_by_dim) {
      return false;
    }
  }

  return true;
}

bool TensorLayout::operator!=(const TensorLayout &t1) const { return !(operator==(t1)); }

bool TensorLayout::IsSameWithoutSplit(const TensorLayout &t1) const {
  if (!IsSameTensorMap(t1) || !IsSameTensorShape(t1)) {
    return false;
  }
  auto first_array = tensor_map().array();
  auto second_array = t1.tensor_map().array();
  auto first_pos = std::find_if(first_array.begin(), first_array.end(), [&](const int64_t ele) { return ele != -1; });
  auto second_pos =
    std::find_if(second_array.begin(), second_array.end(), [&](const int64_t ele) { return ele != -1; });
  if (first_pos != first_array.end() || second_pos != second_array.end()) {
    return false;
  }
  return true;
}

/*
 * remove elements equal to 1 in tensor_shape, if all elements are 1, squeeze the tensor_shape to [ 1 ]
 * example 1:
 *  original tensor layout:
 *    device arrangement = [ 8 ]
 *    tensor map = [ 0 -1 -1 -1 ]
 *    tensor shape = [ 128 64 1 1 ]
 *  return tensor layout:
 *    device arrangement = [ 8 ]
 *    tensor map = [ 0 -1 ]
 *    tensor shape = [ 128 64 ]
 *
 * example 2:
 *  original tensor layout:
 *    device arrangement = [ 8 ]
 *    tensor map = [ -1 -1 -1 -1 ]
 *    tensor shape = [ 1 1 1 1 ]
 *  return tensor layout:
 *    device arrangement = [ 8 ]
 *    tensor map = [ -1 ]
 *    tensor shape = [ 1 ]
 */
TensorLayout TensorLayout::SqueezeShape() const {
  TensorLayout out;
  Map out_map;
  Arrangement out_shape;
  if (tensor_shape_.size() == 1) {
    (void)out_map.Init({MAP_NONE});
    (void)out_shape.Init({1});
    (void)out.Init(device_arrangement_, out_map, out_shape);
    return out;
  }
  std::vector<size_t> squeeze_list = tensor_shape_.GetSqueezeIdx();
  if (!tensor_map_.CheckNoneByIdxList(squeeze_list)) {
    MS_LOG(ERROR) << "CheckNoneByIdxList failed, this may not happen under current situation";
    return *this;
  }
  out_shape = tensor_shape_.GetSqueezeArrangement();
  out_map = tensor_map_.SqueezeMapByIdxList(squeeze_list);
  (void)out.Init(device_arrangement_, out_map, out_shape);
  return out;
}

TensorLayout TensorLayout::TransferRepeatLayout() const {
  Shape dev_mat(device_arrangement_origin_.array());
  Shape tensor_map(tensor_map_origin_.GetDimSize(), -1);
  Shape tensor_shape(tensor_shape_origin_.array());
  TensorLayout repeat;
  if (repeat.InitFromVector(dev_mat, tensor_map, tensor_shape) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Init from vector failed.";
  }
  return repeat;
}

// Generate a totally shard tensor slice shape for parallel optimizer
Status TensorLayout::GenerateOptShardSliceShape() {
  MS_LOG(INFO) << "layout for GetOptShardSliceShape is " << StandardToString();
  Shape dev_max = device_arrangement_.array();
  Shape tensor_map = tensor_map_.array();
  Shape repeated_dev;
  for (size_t i = 0; i < dev_max.size(); i++) {
    if (tensor_map_.GetIndexByValue(static_cast<int64_t>(i)) == MAP_NONE) {
      repeated_dev.push_back(dev_max[dev_max.size() - 1 - i]);
      dev_max[dev_max.size() - 1 - i] = 1;
    }
  }
  if (repeated_dev.empty()) {
    MS_LOG(INFO) << "Tensor is totally shard already.";
    return Status::FAILED;
  }
  int64_t repeated_num =
    std::accumulate(repeated_dev.begin(), repeated_dev.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  int64_t split_num;
  int64_t optimizer_weight_shard_size = ParallelContext::GetInstance()->optimizer_weight_shard_size();
  if (optimizer_weight_shard_size != -1) {
    repeated_num = optimizer_weight_shard_size;
  }
  if (tensor_map[0] == MAP_NONE) {
    split_num = repeated_num;
  } else {
    split_num = dev_max[dev_max.size() - 1 - static_cast<size_t>(tensor_map[0])] * repeated_num;
  }
  if (tensor_shape_.array()[0] % split_num != 0) {
    MS_LOG(INFO) << "Tensor could not be shard on the first dimension.";
    return Status::FAILED;
  }
  Shape origin_slice_shape = slice_shape().array();
  origin_slice_shape[0] = tensor_shape_.array()[0] / split_num;
  opt_shard_slice_shape_ = origin_slice_shape;
  return Status::SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
