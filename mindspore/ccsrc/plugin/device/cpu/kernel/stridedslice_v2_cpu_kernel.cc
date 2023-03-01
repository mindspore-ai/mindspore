/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/stridedslice_v2_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include <complex>
#include <unordered_map>
#include <map>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kStridedSliceV2InputsNum = 1;
constexpr size_t kStridedSliceV2DynamicInputsNum = 4;
constexpr size_t kStridedSliceV2OutputsNum = 1;

void PadStridedSliceV2ParameterTo8D(StridedSliceParameter *param) {
  int32_t begins[DIMENSION_8D];
  int32_t ends[DIMENSION_8D];
  int32_t strides[DIMENSION_8D];
  int32_t input_shape[DIMENSION_8D];
  int32_t i;
  for (i = 0; i < param->num_axes_; ++i) {
    begins[i] = param->begins_[i];
    ends[i] = MSMIN(param->ends_[i], param->in_shape_[i]);
    strides[i] = param->strides_[i];
    input_shape[i] = param->in_shape_[i];
  }
  for (i = param->num_axes_; i < param->in_shape_length_; ++i) {
    input_shape[i] = param->in_shape_[i];
    begins[i] = 0;
    ends[i] = param->in_shape_[i];
    strides[i] = 1;
  }
  int32_t real_index = param->in_shape_length_ - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param->begins_[i] = begins[real_index];
      param->ends_[i] = ends[real_index];
      param->strides_[i] = strides[real_index];
      param->in_shape_[i] = input_shape[real_index--];
    } else {
      param->begins_[i] = 0;
      param->ends_[i] = 1;
      param->strides_[i] = 1;
      param->in_shape_[i] = 1;
    }
  }
  param->num_axes_ = DIMENSION_8D;
  param->in_shape_length_ = DIMENSION_8D;
}

bool LoopContinue(int stride, int i, int end) { return stride > 0 ? i < end : i > end; }

template <typename T>
int DoStridedSliceV2Com(const void *in_data, void *out_data, StridedSliceParameter *param) {
  if (in_data == nullptr || out_data == nullptr || param == nullptr) {
    return NNACL_NULL_PTR;
  }
  if (param->num_axes_ > DIMENSION_8D) {
    return NNACL_PARAM_INVALID;
  }
  int *begins = param->begins_;
  int *ends = param->ends_;
  int *strides = param->strides_;
  int *in_shape = param->in_shape_;
  if (param->num_axes_ < DIMENSION_8D) {
    PadStridedSliceV2ParameterTo8D(param);
  }
  int dim_offset[DIMENSION_8D - 1];
  dim_offset[kIndex6] = in_shape[kIndex7];
  dim_offset[kIndex5] = in_shape[kIndex6] * dim_offset[kIndex6];
  dim_offset[kIndex4] = in_shape[kIndex5] * dim_offset[kIndex5];
  dim_offset[kIndex3] = in_shape[kIndex4] * dim_offset[kIndex4];
  dim_offset[kIndex2] = in_shape[kIndex3] * dim_offset[kIndex3];
  dim_offset[1] = in_shape[kIndex2] * dim_offset[kIndex2];
  dim_offset[0] = in_shape[1] * dim_offset[1];
  size_t out_offset = 0;
  int32_t dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7;
  for (dim0 = begins[0]; LoopContinue(strides[0], dim0, ends[0]); dim0 += strides[0]) {
    for (dim1 = begins[1]; LoopContinue(strides[1], dim1, ends[1]); dim1 += strides[1]) {
      for (dim2 = begins[kIndex2]; LoopContinue(strides[kIndex2], dim2, ends[kIndex2]); dim2 += strides[kIndex2]) {
        for (dim3 = begins[kIndex3]; LoopContinue(strides[kIndex3], dim3, ends[kIndex3]); dim3 += strides[kIndex3]) {
          for (dim4 = begins[kIndex4]; LoopContinue(strides[kIndex4], dim4, ends[kIndex4]); dim4 += strides[kIndex4]) {
            for (dim5 = begins[kIndex5]; LoopContinue(strides[kIndex5], dim5, ends[kIndex5]);
                 dim5 += strides[kIndex5]) {
              for (dim6 = begins[kIndex6]; LoopContinue(strides[kIndex6], dim6, ends[kIndex6]);
                   dim6 += strides[kIndex6]) {
                for (dim7 = begins[kIndex7]; LoopContinue(strides[kIndex7], dim7, ends[kIndex7]);
                     dim7 += strides[kIndex7]) {
                  int32_t in_offset = dim0 * dim_offset[0] + dim1 * dim_offset[1] + dim2 * dim_offset[kIndex2] +
                                      dim3 * dim_offset[kIndex3] + dim4 * dim_offset[kIndex4] +
                                      dim5 * dim_offset[kIndex5] + dim6 * dim_offset[kIndex6] + dim7;
                  auto out_ptr = static_cast<T *>(out_data);
                  auto int_ptr = static_cast<const T *>(in_data);
                  out_ptr[out_offset] = int_ptr[in_offset];
                  out_offset++;
                }
              }
            }
          }
        }
      }
    }
  }
  return NNACL_OK;
}

static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
  {kStridedSliceV2,
   {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    KernelAttr()
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeBool),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex128),
    KernelAttr()
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeBool),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeUInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeComplex128)}}};

template <typename T>
void ParseStrideSliceMasksST(const BaseOperatorPtr &base_operator, std::vector<T> *begin, std::vector<T> *end,
                             std::vector<T> *stride, const ShapeVector &input_shape, size_t shape_dim_input,
                             size_t slice_len) {
  std::vector<T> &_begin_attr = *begin;
  std::vector<T> &_end_attr = *end;
  std::vector<T> &_stride_attr = *stride;
  auto prim = base_operator->GetPrim();
  auto begin_mask_int = GetValue<int64_t>(prim->GetAttr(kAttrBeginMask));
  auto begin_mask = Dec2Bin(begin_mask_int);
  auto end_mask_int = GetValue<int64_t>(prim->GetAttr(kAttrEndMask));
  auto end_mask = Dec2Bin(end_mask_int);
  auto ellipsis_mask_int = GetValue<int64_t>(prim->GetAttr(kAttrEllipsisMask));
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
  auto new_axis_mask_int = GetValue<int64_t>(prim->GetAttr(kAttrNewAxisMask));
  auto new_axis_mask = Dec2Bin(new_axis_mask_int);
  auto shrink_axis_mask_int = GetValue<int64_t>(prim->GetAttr(kAttrShrinkAxisMask));
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
  size_t i = 0;
  size_t j = 0;
  std::vector<T> begin_new;
  std::vector<T> end_new;
  std::vector<T> stride_new;
  while (i < shape_dim_input || j < slice_len) {
    T begin_j;
    T end_j;
    T stride_j;
    if (j < slice_len) {
      begin_j = _begin_attr[j];
      end_j = _end_attr[j];
      stride_j = _stride_attr[j];
      if (begin_mask[j]) {
        begin_j = _stride_attr[j] < 0 ? static_cast<T>(input_shape[i]) - 1 : 0;
      }
      if (end_mask[j]) {
        end_j = _stride_attr[j] < 0 ? -1 : static_cast<T>(input_shape[i]);
      }
      if (ellipsis_mask[j]) {
        begin_j = 0;
        end_j = static_cast<T>(input_shape[i]);
        stride_j = 1;
      }
      if (new_axis_mask[j]) {
        j++;
        continue;
      }
      if (shrink_axis_mask[j]) {
        end_j = _begin_attr[j] + 1;
        stride_j = 1;
      }
      if (end_j > input_shape[i]) {
        end_j = input_shape[i];
      }
    } else {
      begin_j = 0;
      end_j = static_cast<T>(input_shape[i]);
      stride_j = 1;
    }
    begin_new.push_back(begin_j);
    end_new.push_back(end_j);
    stride_new.push_back(stride_j);
    i++;
    j++;
  }
  _begin_attr.assign(begin_new.begin(), begin_new.end());
  _end_attr.assign(end_new.begin(), end_new.end());
  _stride_attr.assign(stride_new.begin(), stride_new.end());
}

template <typename T>
void FillEmptyDimsST(const BaseOperatorPtr &base_operator, std::vector<T> *begin, std::vector<T> *end,
                     std::vector<T> *stride, ShapeVector *input_shape) {
  std::vector<T> &_begin = *begin;
  std::vector<T> &_end = *end;
  std::vector<T> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << base_operator->name()
                      << "', the length of 'begin', 'stride' and 'end' should be equal "
                         "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                      << _begin.size() << ", the length of 'stride': " << _stride.size()
                      << ", the length of 'end': " << _end.size()
                      << ", the dimension of 'input_x': " << _input_shape.size();
  }
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i >= _input_shape.size()) {
      _input_shape.push_back(1);
    }
    if (i < _begin.size()) {
      T dim = static_cast<T>(_input_shape[i]);
      _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<T>(0)) : _begin[i], dim - 1);
    } else {
      _begin.push_back(0);
    }

    if (i >= _end.size()) {
      _end.push_back(i < _input_shape.size() ? static_cast<T>(_input_shape[i]) : 1);
    }

    if (i >= _stride.size()) {
      _stride.push_back(1);
    }
  }
}
}  // namespace

bool StridedSliceV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  base_operator_ = base_operator;
  kernel_name_ = base_operator->name();

  if (inputs.size() != kStridedSliceV2InputsNum && inputs.size() != kStridedSliceV2DynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kStridedSliceV2InputsNum
                      << " or " << kStridedSliceV2DynamicInputsNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceV2OutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  dtype_ = inputs[kIndex0]->GetDtype();
  dtype_attr_ = inputs[kIndex1]->GetDtype();

  return true;
}

int StridedSliceV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetShapeVector();
  begin_shape_ = inputs[kIndex1]->GetShapeVector();
  end_shape_ = inputs[kIndex2]->GetShapeVector();
  stride_shape_ = inputs[kIndex3]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();

  if (inputs.size() == kStridedSliceV2DynamicInputsNum) {
    return KRET_OK;
  }

  auto prim = base_operator->GetPrim();
  auto begin = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrBegin));
  auto end = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrEnd));
  auto stride = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrStrides));
  InitSliceParam<int64_t>(base_operator, &begin, &end, &stride);

  parallel_ = MatchParallelPattern();
  if (parallel_) {
    InitParallelParam();
  }
  return KRET_OK;
}

bool StridedSliceV2CpuKernelMod::MatchParallelPattern() {
  // This function is seeking if that the number of only one dimension
  // is different between input and output. If so, we can do some trick.
  // Example 1:
  // input shape info:  [1, 80, 46, 40]
  // output shape info: [1, 80, 20, 40]
  // Example 2:
  // input shape info:  [1, 46, 40]
  // output shape info: [1, 20, 40]
  if (input_shape_.size() == output_shape_.size()) {
    std::vector<int> axis_list;
    for (size_t i = 0; i < input_shape_.size(); ++i) {
      if (input_shape_[i] != output_shape_[i]) {
        (void)axis_list.emplace_back(i);
      }
    }
    if (axis_list.size() == 1) {
      split_axis_ = axis_list.front();
      return true;
    }
  }
  return false;
}

void StridedSliceV2CpuKernelMod::InitParallelParam() {
  outer_ = SizeToInt(
    std::accumulate(input_shape_.begin(), input_shape_.begin() + split_axis_, size_t(1), std::multiplies<size_t>()));
  inner_ = SizeToInt(
    std::accumulate(input_shape_.begin() + split_axis_ + 1, input_shape_.end(), size_t(1), std::multiplies<size_t>()));

  int max_thread_num = SizeToInt(common::ThreadPool::GetInstance().GetSyncRunThreadNum());
  int thread_num = 1;
  if (outer_ == 1) {
    parallel_strategy_ = kOnSplitAxis;
    thread_num = std::min(SizeToInt(output_shape_[split_axis_]), max_thread_num);
    if (thread_num == 0) {
      slice_param_.op_parameter_.thread_num_ = 1;
      return;
    }
    cal_num_per_thread_ = UP_DIV(output_shape_[split_axis_], thread_num);
  } else {
    parallel_strategy_ = kOnOuter;
    thread_num = std::min(outer_, max_thread_num);
    if (thread_num == 0) {
      slice_param_.op_parameter_.thread_num_ = 1;
      return;
    }
    cal_num_per_thread_ = UP_DIV(outer_, thread_num);
  }
  slice_param_.op_parameter_.thread_num_ = thread_num;
}

template <typename T>
void StridedSliceV2CpuKernelMod::InitSliceParam(const BaseOperatorPtr &base_operator, std::vector<T> *begin,
                                                std::vector<T> *end, std::vector<T> *stride) {
  static const std::unordered_map<TypeId, std::pair<TypeIdC, int>> type_convert_map = {
    {kNumberTypeBool, {::kNumberTypeBool, sizeof(bool)}},
    {kNumberTypeInt8, {::kNumberTypeInt8, sizeof(int8_t)}},
    {kNumberTypeInt16, {::kNumberTypeInt16, sizeof(int16_t)}},
    {kNumberTypeInt32, {::kNumberTypeInt32, sizeof(int32_t)}},
    {kNumberTypeInt64, {::kNumberTypeInt64, sizeof(int64_t)}},
    {kNumberTypeUInt8, {::kNumberTypeUInt8, sizeof(uint8_t)}},
    {kNumberTypeUInt16, {::kNumberTypeUInt16, sizeof(uint16_t)}},
    {kNumberTypeUInt32, {::kNumberTypeUInt32, sizeof(uint32_t)}},
    {kNumberTypeUInt64, {::kNumberTypeUInt64, sizeof(uint64_t)}},
    {kNumberTypeFloat16, {::kNumberTypeFloat16, sizeof(float16)}},
    {kNumberTypeFloat32, {::kNumberTypeFloat32, sizeof(float)}},
    {kNumberTypeFloat64, {::kNumberTypeFloat64, sizeof(double)}},
    {kNumberTypeComplex64, {::kNumberTypeComplex64, sizeof(complex64)}},
    {kNumberTypeComplex128, {::kNumberTypeComplex64, sizeof(complex128)}}};

  auto type_pair = type_convert_map.find(dtype_);
  if (type_pair == type_convert_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' must be bool, int8, int16, int32, int64, float16, float32, "
                         "float64, uint8, uint16, uint32, uint64, complex64 or complex128, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  data_size_ = type_pair->second.second;
  slice_param_.data_type = type_pair->second.first;
  auto input_shape_pad = input_shape_;
  shape_dim_input = input_shape_.size();
  FillEmptyDimsST<T>(base_operator, begin, end, stride, &input_shape_pad);
  ParseStrideSliceMasksST<T>(base_operator, begin, end, stride, input_shape_, shape_dim_input, slice_len);
  FillEmptyDimsST<T>(base_operator, begin, end, stride, &input_shape_pad);

  std::vector<T> &_begin = *begin;
  std::vector<T> &_end = *end;
  std::vector<T> &_stride = *stride;
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    slice_param_.in_shape_[i] = SizeToInt(input_shape_pad[i]);
    if (dtype_attr_ == kNumberTypeInt64) {
      slice_param_.begins_[i] = LongToInt(_begin[i]);
      slice_param_.ends_[i] = LongToInt(_end[i]);
      slice_param_.strides_[i] = LongToInt(_stride[i]);
    } else {
      slice_param_.begins_[i] = _begin[i];
      slice_param_.ends_[i] = _end[i];
      slice_param_.strides_[i] = _stride[i];
    }
  }
  slice_param_.in_shape_length_ = DIMENSION_8D;
  slice_param_.num_axes_ = DIMENSION_8D;
}

common::Status StridedSliceV2CpuKernelMod::RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr,
                                                          int start_pos) {
  int begin_index = slice_param_.begins_[split_axis_];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * input_shape_[split_axis_] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * output_shape_[split_axis_] * inner_size;
  int cur_outer = outer_ - start_pos;
  if (cur_outer <= 0) {
    return common::SUCCESS;
  }
  cur_outer = cur_outer > cal_num_per_thread_ ? cal_num_per_thread_ : cur_outer;
  FastStride(cur_in_ptr, cur_out_ptr, output_shape_[split_axis_], slice_param_.strides_[split_axis_], cur_outer,
             inner_size, input_shape_[split_axis_] * inner_size);
  return common::SUCCESS;
}

common::Status StridedSliceV2CpuKernelMod::RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr,
                                                              int start_pos) {
  int begin_index = slice_param_.begins_[split_axis_];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * slice_param_.strides_[split_axis_] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * inner_size;
  int cal_axis_num = output_shape_[split_axis_] - start_pos;
  if (cal_axis_num <= 0) {
    return common::SUCCESS;
  }
  cal_axis_num = cal_axis_num > cal_num_per_thread_ ? cal_num_per_thread_ : cal_axis_num;
  FastStride(cur_in_ptr, cur_out_ptr, cal_axis_num, slice_param_.strides_[split_axis_], 1, inner_size, 0);
  return common::SUCCESS;
}

void StridedSliceV2CpuKernelMod::ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num) {
  int thread_index = 0;
  std::vector<common::Task> tasks;
  std::function<common::Status(StridedSliceV2CpuKernelMod *, const uint8_t *, uint8_t *, int)> execute_func;
  if (parallel_strategy_ == kOnOuter) {
    execute_func = &StridedSliceV2CpuKernelMod::RunTaskOnOuter;
  } else if (parallel_strategy_ == kOnSplitAxis) {
    execute_func = &StridedSliceV2CpuKernelMod::RunTaskOnSplitAxis;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', not supports parallel execute strategy.";
  }

  while (thread_index < thread_num) {
    (void)tasks.emplace_back(
      std::bind(execute_func, this, input_addr, output_addr, thread_index * cal_num_per_thread_));
    thread_index++;
  }
  ParallelLaunch(tasks);
}

template <typename T>
void StridedSliceV2CpuKernelMod::StridedSliceV2LaunchDynamicType(const std::vector<kernel::AddressPtr> &inputs) {
  if (begin_shape_.size() != 1 || end_shape_.size() != 1 || stride_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'begin', 'end', 'strides' should be equal "
                         "to 1, but got the dimension of 'begin': "
                      << begin_shape_.size() << ", the dimension of 'end': " << end_shape_.size()
                      << ", and the dimension of 'strides': " << stride_shape_.size();
  }
  auto begin_ptr = static_cast<T *>(inputs[1]->addr);
  auto end_ptr = static_cast<T *>(inputs[2]->addr);
  auto strides_ptr = static_cast<T *>(inputs[3]->addr);
  std::vector<T> begin{begin_ptr, begin_ptr + begin_shape_[0]};
  std::vector<T> end{end_ptr, end_ptr + end_shape_[0]};
  std::vector<T> stride{strides_ptr, strides_ptr + stride_shape_[0]};
  slice_len = begin.size();
  InitSliceParam<T>(base_operator_, &begin, &end, &stride);
}

void StridedSliceV2CpuKernelMod::StridedSliceV2LaunchCal(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  // for begin, end, stride are not const input
  if (dtype_attr_ == kNumberTypeInt32) {
    StridedSliceV2LaunchDynamicType<int32_t>(inputs);
  } else {
    StridedSliceV2LaunchDynamicType<int64_t>(inputs);
  }
  parallel_ = MatchParallelPattern();
  if (parallel_) {
    InitParallelParam();
  }
}

bool StridedSliceV2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> & /* workspace */,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() == kStridedSliceV2DynamicInputsNum) {
    StridedSliceV2LaunchCal(inputs, outputs);
  }
  auto input_addr = static_cast<uint8_t *>(inputs[0]->addr);
  auto output_addr = static_cast<uint8_t *>(outputs[0]->addr);

  int thread_std = 2;
  int thread_num = slice_param_.op_parameter_.thread_num_;
  if (parallel_ && thread_num >= thread_std) {
    ParallelRun(input_addr, output_addr, thread_num);
  } else if (dtype_ == kNumberTypeComplex128) {
    (void)DoStridedSliceV2Com<complex128>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeComplex64) {
    (void)DoStridedSliceV2Com<complex64>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeFloat16) {
    (void)DoStridedSliceV2Com<float16>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeFloat32) {
    (void)DoStridedSliceV2Com<float>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeFloat64) {
    (void)DoStridedSliceV2Com<double>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeInt8) {
    (void)DoStridedSliceV2Com<int8_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeInt16) {
    (void)DoStridedSliceV2Com<int16_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeInt32) {
    (void)DoStridedSliceV2Com<int32_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeInt64) {
    (void)DoStridedSliceV2Com<int64_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeUInt8) {
    (void)DoStridedSliceV2Com<uint8_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeUInt16) {
    (void)DoStridedSliceV2Com<uint16_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeUInt32) {
    (void)DoStridedSliceV2Com<uint32_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeUInt64) {
    (void)DoStridedSliceV2Com<uint64_t>(input_addr, output_addr, &slice_param_);
  } else if (dtype_ == kNumberTypeBool) {
    (void)DoStridedSliceV2Com<bool>(input_addr, output_addr, &slice_param_);
  }
  return true;
}

std::vector<KernelAttr> StridedSliceV2CpuKernelMod::GetOpSupport() {
  auto iter = support_list_map.find(kStridedSliceV2);
  return iter->second;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, StridedSliceV2, StridedSliceV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
