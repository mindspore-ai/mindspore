/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ops_utils.h"
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kStridedSliceMaxDims = 8;
}  // namespace

std::vector<bool> Dec2Bin(const int64_t &mask) {
  auto mask_str = std::bitset<kStridedSliceMaxDims>(mask).to_string();
  size_t dim_idx = 0;
  std::vector<bool> result(kStridedSliceMaxDims, false);
  for (auto iter = mask_str.rbegin(); iter != mask_str.rend(); ++iter) {
    if (*iter == '1') {
      result[dim_idx] = true;
    }
    ++dim_idx;
  }
  return result;
}

void FillEmptyDims(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                   std::vector<int64_t> *stride, ShapeVector *input_shape, bool is_gpu_strided) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    auto kernel_name = base_operator->name();
    MS_LOG(EXCEPTION) << "For '" << kernel_name
                      << "', the length of 'begin', 'stride' and 'end' should be equal "
                         "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                      << _begin.size() << ", the length of 'stride': " << _stride.size()
                      << ", the length of 'end': " << _end.size()
                      << ", the dimension of 'input_x': " << _input_shape.size();
  }

  for (size_t i = 0; i < kStridedSliceMaxDims; i++) {
    if (i >= _input_shape.size()) {
      _input_shape.push_back(1);
    }

    if (i < _begin.size()) {
      int64_t dim = _input_shape[i];
      if (is_gpu_strided) {
        // GPU kernel is flattened using offset to get stride slice
        _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<int64_t>(0)) : _begin[i], dim - 1);
      } else {
        // CPU using for begin is larger than end the circle will be break
        _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<int64_t>(0)) : _begin[i], dim);
      }
    } else {
      _begin.push_back(0);
    }

    if (i < _end.size()) {
      int64_t dim = _input_shape[i];
      _end[i] = std::max(_end[i] < 0 ? _end[i] + dim : std::min(_end[i], dim), static_cast<int64_t>(-1));
    } else {
      _end.push_back(i < _input_shape.size() ? _input_shape[i] : 1);
    }

    if (i >= _stride.size()) {
      _stride.push_back(1);
    }
  }
}

void ComputeBeginMask(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                      const std::vector<int64_t> &stride, const ShapeVector &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
  auto begin_mask_int = kernel_ptr->get_begin_mask();
  auto begin_mask = Dec2Bin(begin_mask_int);
  for (size_t i = 0; i < begin_mask.size(); i++) {
    if (i < kStridedSliceMaxDims && begin_mask[i]) {
      _begin[i] = stride[i] < 0 ? input_shape[i] - 1 : 0;
    }
  }
}

void ComputeEndMask(const BaseOperatorPtr &base_operator, std::vector<int64_t> *end, const std::vector<int64_t> &stride,
                    const ShapeVector &input_shape) {
  std::vector<int64_t> &_end = *end;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
  auto end_mask_int = kernel_ptr->get_end_mask();
  auto end_mask = Dec2Bin(end_mask_int);
  for (size_t j = 0; j < end_mask.size(); j++) {
    if (j < kStridedSliceMaxDims && end_mask[j]) {
      _end[j] = stride[j] < 0 ? -1 : input_shape[j];
    }
  }
}

void ComputeEllipsisMask(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                         std::vector<int64_t> *stride, const ShapeVector &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
  auto ellipsis_mask_int = kernel_ptr->get_ellipsis_mask();
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
  for (size_t k = 0; k < ellipsis_mask.size(); k++) {
    if (k < kStridedSliceMaxDims && ellipsis_mask[k]) {
      _begin[k] = 0;
      _end[k] = input_shape[k];
      _stride[k] = 1;
    }
  }
}

void ComputNewAxisMask(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                       std::vector<int64_t> *stride, const ShapeVector &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
  auto new_axis_mask_int = kernel_ptr->get_new_axis_mask();
  auto new_axis_mask = Dec2Bin(new_axis_mask_int);
  for (size_t l = 0; l < new_axis_mask.size(); l++) {
    if (l < kStridedSliceMaxDims && new_axis_mask[l]) {
      _begin[l] = 0;
      _end[l] = input_shape[l];
      _stride[l] = 1;
    }
  }
}

void ComputeShrinkAxisMask(const BaseOperatorPtr &base_operator, const std::vector<int64_t> &begin,
                           std::vector<int64_t> *end, std::vector<int64_t> *stride) {
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
  auto shrink_axis_mask_int = kernel_ptr->get_shrink_axis_mask();
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
  for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
    if (m < kStridedSliceMaxDims && shrink_axis_mask[m]) {
      _end[m] = _end[m] > begin[m] ? begin[m] + 1 : begin[m] - 1;
      _stride[m] = _end[m] > begin[m] ? 1 : -1;
    }
  }
}

void ParseStrideSliceMasks(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                           std::vector<int64_t> *stride, const ShapeVector &input_shape) {
  ComputeBeginMask(base_operator, begin, *stride, input_shape);
  ComputeEndMask(base_operator, end, *stride, input_shape);
  ComputeEllipsisMask(base_operator, begin, end, stride, input_shape);
  ComputNewAxisMask(base_operator, begin, end, stride, input_shape);
  ComputeShrinkAxisMask(base_operator, *begin, end, stride);
}

float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? SizeToFloat(in_size - 1) / static_cast<float>(out_size - 1)
                                         : SizeToFloat(in_size) / static_cast<float>(out_size);
}

float ScaleGrid(const int x, const float scale, bool half_pixel_centers) {
  if (half_pixel_centers) {
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  } else {
    return static_cast<float>(x) * scale;
  }
}

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size, const float scale,
                                 CachedInterpolation *interpolation, bool half_pixel_centers) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ScaleGrid(SizeToInt(i), scale, half_pixel_centers);
    const float in_f = std::floor(in);
    interpolation[i].lower = std::max(static_cast<int64_t>(in_f), static_cast<int64_t>(0));
    interpolation[i].upper = std::min(static_cast<int64_t>(std::ceil(in)), static_cast<int64_t>(in_size - 1));
    interpolation[i].lerp = in - in_f;
  }
}

void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                     const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape) {
  if (start.size() != stop.size() || start.size() != step.size() || start.size() > input_shape.size()) {
    MS_LOG(EXCEPTION)
      << "TensorCopySlices requires the length of begin, stride and end must be equal and less than input dimension.";
  }

  size_t size = start.size();
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] <= start[i]) {
      MS_LOG(EXCEPTION) << "Invalid slice: (" << start[i] << ", " << stop[i] << " ," << step[i] << ")";
    }
    // Operator need to be generalized in the future. Only support to copy continuous memory now.
    if (step[i] != 1) {
      MS_LOG(EXCEPTION) << "The element in step only support 1, but got:" << step;
    }
  }

  size_t slice_pos = size;
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] - start[i] > 1) {
      slice_pos = i;
      break;
    }
  }

  for (size_t i = slice_pos + 1; i < size; ++i) {
    if (stop[i] - start[i] != input_shape[i]) {
      MS_LOG(EXCEPTION) << "Only support copy continuous memory now. For example tensor[0, 0:100] is fine, "
                           "but tensor[0:100, 0] is not supported.";
    }
  }
}

size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                   const std::vector<int64_t> &stop) {
  for (size_t i = 0; i < start.size(); ++i) {
    if (stop[i] - start[i] != 1) {
      return SizetMulWithOverflowCheck(LongToSize(stop[i] - start[i]), LongToSize(dim_offset[i]));
    }
  }
  return LongToSize(dim_offset[start.size() - 1]);
}

std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape) {
  std::vector<int64_t> dim_offset;
  int64_t offset = 1;
  for (auto iter = input_shape.rbegin(); iter != input_shape.rend(); ++iter) {
    dim_offset.push_back(offset);
    offset = offset * (*iter);
  }
  std::reverse(dim_offset.begin(), dim_offset.end());
  return dim_offset;
}

size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                 const std::vector<int64_t> &dim_offset) {
  size_t size = start.size();
  size_t offset = 0;
  for (size_t i = 0; i < size; ++i) {
    offset += SizetMulWithOverflowCheck(LongToSize(dim_offset[i]), LongToSize(start[i]));
    if (stop[i] - start[i] != 1) {
      break;
    }
  }
  return offset;
}

std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> GetAlignments(const std::string &alignment) {
  static const mindspore::HashMap<std::string, std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment>> AlignmentMap{
    {"RIGHT_LEFT", {MatrixDiag::RIGHT, MatrixDiag::LEFT}},
    {"LEFT_RIGHT", {MatrixDiag::LEFT, MatrixDiag::RIGHT}},
    {"RIGHT_RIGHT", {MatrixDiag::RIGHT, MatrixDiag::RIGHT}},
    {"LEFT_LEFT", {MatrixDiag::LEFT, MatrixDiag::LEFT}}};

  auto alignment_iter = AlignmentMap.find(alignment);
  if (alignment_iter == AlignmentMap.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For  current kernel, input alignment is invalid: " << alignment
                               << ". please limit it to {RIGHT_LEFT, LEFT_RIGHT, RIGHT_RIGHT, LEFT_LEFT}";
  }
  return alignment_iter->second;
}

namespace broadcast_utils {
bool AlignedBroadCastShape(size_t align_rank, std::vector<size_t> *broadcast, std::vector<size_t> *lhs,
                           std::vector<size_t> *rhs) {
  if (broadcast == nullptr || lhs == nullptr || rhs == nullptr) {
    MS_LOG(ERROR) << "input is nullptr.";
    return false;
  }
  size_t broadcast_rank = broadcast->size();
  size_t l_rank = lhs->size();
  size_t r_rank = rhs->size();
  if (broadcast_rank > align_rank || l_rank > align_rank || r_rank > align_rank) {
    return false;
  }
  std::vector<size_t> aligned_broadcast(align_rank, 1);
  std::vector<size_t> aligned_lhs(align_rank, 1);
  std::vector<size_t> aligned_rhs(align_rank, 1);
  size_t broadcast_offset = align_rank - broadcast_rank;
  for (size_t i = 0; i < broadcast_rank; i++) {
    aligned_broadcast[i + broadcast_offset] = (*broadcast)[i];
  }

  size_t l_offset = align_rank - l_rank;
  for (size_t i = 0; i < l_rank; i++) {
    aligned_lhs[i + l_offset] = (*lhs)[i];
  }
  size_t r_offset = align_rank - r_rank;
  for (size_t i = 0; i < r_rank; i++) {
    aligned_rhs[i + r_offset] = (*rhs)[i];
  }
  *broadcast = aligned_broadcast;
  *lhs = aligned_lhs;
  *rhs = aligned_rhs;
  return true;
}
}  // namespace broadcast_utils
}  // namespace kernel
}  // namespace mindspore
