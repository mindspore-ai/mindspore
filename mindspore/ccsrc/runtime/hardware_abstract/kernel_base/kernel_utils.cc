/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"
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

void FillEmptyDims(const std::string &kernel_name, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                   std::vector<int64_t> *stride, ShapeVector *input_shape, bool is_gpu_strided) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
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

void ComputeBeginMask(std::vector<int64_t> *begin, const std::vector<int64_t> &stride, const ShapeVector &input_shape,
                      KernelTensor *kernel_tensor) {
  std::vector<int64_t> &_begin = *begin;
  auto begin_mask_int = kernel_tensor->GetValueWithCheck<int64_t>();
  auto begin_mask = Dec2Bin(begin_mask_int);
  for (size_t i = 0; i < begin_mask.size(); i++) {
    if (i < kStridedSliceMaxDims && begin_mask[i]) {
      _begin[i] = stride[i] < 0 ? input_shape[i] - 1 : 0;
    }
  }
}

void ComputeEndMask(std::vector<int64_t> *end, const std::vector<int64_t> &stride, const ShapeVector &input_shape,
                    KernelTensor *kernel_tensor) {
  std::vector<int64_t> &_end = *end;
  auto end_mask_int = kernel_tensor->GetValueWithCheck<int64_t>();
  auto end_mask = Dec2Bin(end_mask_int);
  for (size_t j = 0; j < end_mask.size(); j++) {
    if (j < kStridedSliceMaxDims && end_mask[j]) {
      _end[j] = stride[j] < 0 ? -1 : input_shape[j];
    }
  }
}

void ComputeEllipsisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                         const ShapeVector &input_shape, KernelTensor *kernel_tensor) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto ellipsis_mask_int = kernel_tensor->GetValueWithCheck<int64_t>();
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
  for (size_t k = 0; k < ellipsis_mask.size(); k++) {
    if (k < kStridedSliceMaxDims && ellipsis_mask[k]) {
      _begin[k] = 0;
      _end[k] = input_shape[k];
      _stride[k] = 1;
    }
  }
}

void ComputNewAxisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                       const ShapeVector &input_shape, KernelTensor *kernel_tensor) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto new_axis_mask_int = kernel_tensor->GetValueWithCheck<int64_t>();
  auto new_axis_mask = Dec2Bin(new_axis_mask_int);
  for (size_t l = 0; l < new_axis_mask.size(); l++) {
    if (l < kStridedSliceMaxDims && new_axis_mask[l]) {
      _begin[l] = 0;
      _end[l] = input_shape[l];
      _stride[l] = 1;
    }
  }
}

void ComputeShrinkAxisMask(const std::vector<int64_t> &begin, std::vector<int64_t> *end, std::vector<int64_t> *stride,
                           KernelTensor *kernel_tensor) {
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto shrink_axis_mask_int = kernel_tensor->GetValueWithCheck<int64_t>();
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
  for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
    if (m < kStridedSliceMaxDims && shrink_axis_mask[m]) {
      _end[m] = _end[m] > begin[m] ? begin[m] + 1 : begin[m] - 1;
      _stride[m] = _end[m] > begin[m] ? 1 : -1;
    }
  }
}

void ParseStrideSliceMasks(const std::vector<kernel::KernelTensor *> &inputs, std::vector<int64_t> *begin,
                           std::vector<int64_t> *end, std::vector<int64_t> *stride, const ShapeVector &input_shape) {
  ComputeBeginMask(begin, *stride, input_shape, inputs[kIndex4]);
  ComputeEndMask(end, *stride, input_shape, inputs[kIndex5]);
  ComputeEllipsisMask(begin, end, stride, input_shape, inputs[kIndex6]);
  ComputNewAxisMask(begin, end, stride, input_shape, inputs[kIndex7]);
  ComputeShrinkAxisMask(*begin, end, stride, inputs[kIndex8]);
}

// ===========================Old interface==========================================================
void FillEmptyDims(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                   std::vector<int64_t> *stride, ShapeVector *input_shape, bool is_gpu_strided) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << base_operator->name()
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

float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? SizeToFloat(in_size - 1) / static_cast<float>(out_size - 1)
                                         : SizeToFloat(in_size) / static_cast<float>(out_size);
}
}  // namespace kernel
}  // namespace mindspore
