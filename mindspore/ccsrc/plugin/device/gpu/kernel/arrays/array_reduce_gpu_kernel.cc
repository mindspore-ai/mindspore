/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/array_reduce_gpu_kernel.h"
#include <memory>
#include "mindspore/core/ops/math_ops.h"
#include "ops/reduce.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/reduce_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim3 = 3;
}  // namespace
template <typename T>
using Complex = mindspore::utils::Complex<T>;
constexpr auto kReduceMean = "ReduceMean";
constexpr auto kReduceMax = "ReduceMax";
constexpr auto kReduceSum = "ReduceSum";
constexpr auto kReduceMin = "ReduceMin";
constexpr auto kReduceProd = "ReduceProd";
constexpr auto kReduceAll = "ReduceAll";
constexpr auto kReduceAny = "ReduceAny";

constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;

const std::map<std::string, ReduceType_t> kReduceTypeMap = {
  {"ReduceMax", ReduceMax}, {"ReduceMean", ReduceMean}, {"ReduceSum", ReduceSum},   {"ReduceMin", ReduceMin},
  {"ReduceAny", ReduceAny}, {"ReduceAll", ReduceAll},   {"ReduceProd", ReduceProd},
};

#define REDUCE_REGISTER(INPUTX, AXIS, T) \
  KernelAttr().AddInputAttr(INPUTX).AddInputAttr(AXIS).AddOutputAttr(INPUTX), &ArrayReduceGpuKernelMod::LaunchKernel<T>

#define REDUCE_REGISTER_COMPLEX(INPUTX, AXIS, T)                              \
  KernelAttr().AddInputAttr(INPUTX).AddInputAttr(AXIS).AddOutputAttr(INPUTX), \
    &ArrayReduceGpuKernelMod::LaunchComplexKernel<T>

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::all_any_list_ = {
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)},
};

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::prod_list_ = {
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, uint64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, uint64_t)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::sum_list_ = {
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, uint64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, uint64_t)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::max_min_list_ = {
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
  {REDUCE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, uint32_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, uint64_t)},
  {REDUCE_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, uint64_t)},
};

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::mean_list_ = {
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER_COMPLEX(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};

std::map<std::string, std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>>>
  ArrayReduceGpuKernelMod::kernel_attr_list_ = {
    {prim::kPrimReduceSum->name(), sum_list_},     {prim::kPrimReduceMean->name(), mean_list_},
    {prim::kPrimReduceProd->name(), prod_list_},   {prim::kPrimReduceMax->name(), max_min_list_},
    {prim::kPrimReduceMin->name(), max_min_list_}, {prim::kPrimReduceAll->name(), all_any_list_},
    {prim::kPrimReduceAny->name(), all_any_list_}};

std::vector<KernelAttr> ArrayReduceGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list_.find(kernel_type_);
  if (iter == kernel_attr_list_.end()) {
    MS_LOG(ERROR) << "For 'Reduce ops', it does not support " << kernel_type_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc> &pair) { return pair.first; });
  return support_list;
}

bool ArrayReduceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }

  auto iter = kernel_attr_list_.find(kernel_type_);
  if (iter == kernel_attr_list_.end()) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc> &pair) { return pair.first; });

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " does not support this data type: " << kernel_attr;
  }
  kernel_func_ = kernel_attr_list_[kernel_type_][index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  InferArrayReduceType();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Reduce>(base_operator);
  keep_dims_ = kernel_ptr->get_keep_dims();
  skip_mode_ = kernel_ptr->get_skip_mode();
  return true;
}

void ArrayReduceGpuKernelMod::FormatAxis(const size_t dims, const std::vector<int> &axis, std::vector<bool> *bitmap) {
  if (dims == 0) {
    return;
  }
  const int dims_len = SizeToInt(dims);
  auto axis_fill = axis;
  if (axis.empty()) {
    for (int i = 0; i < dims_len; i++) {
      axis_fill.push_back(i);
    }
  }
  for (size_t i = 0; i < axis_fill.size(); i++) {
    int index = axis_fill[i];
    if (index < -dims_len || index >= dims_len) {
      MS_LOG(EXCEPTION) << "Invalid reduction dimension (" << index << " for input with " << dims << " dimension(s)";
    }
    index = (index + dims) % dims;
    if ((*bitmap)[index]) {
      MS_LOG(EXCEPTION) << "Invalid reduction arguments: Axes contains duplicate dimension: " << index;
    }
    (*bitmap)[index] = true;
  }
}

void ArrayReduceGpuKernelMod::SimplyReduce(const ShapeVector &input_shape, const std::vector<int> &axis) {
  std::vector<bool> bitmap(input_shape.size(), false);
  FormatAxis(input_shape.size(), axis, &bitmap);
  size_t dim_index = 0;
  for (; dim_index < input_shape.size(); dim_index++) {
    if (input_shape[dim_index] != 1) break;
  }
  if (dim_index >= input_shape.size()) {
    reduce_first_axis_ = true;
  } else {
    input_reshape_.clear();
    reduce_first_axis_ = bitmap[dim_index];
    input_reshape_.push_back(input_shape[dim_index]);
    dim_index++;
    for (; dim_index < input_shape.size(); dim_index++) {
      const size_t size = input_shape[dim_index];
      if (size == 1) {
        bitmap[dim_index] = bitmap[dim_index - 1];
      }
      if (bitmap[dim_index] != bitmap[dim_index - 1]) {
        input_reshape_.push_back(size);
      } else {
        input_reshape_.back() *= size;
      }
    }
  }
}

int ArrayReduceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  need_skip_execute_ = false;
  all_match_ = false;
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto inputA_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_num_ = SizeOf(inputA_shape);

  std::vector<int64_t> attr_axis;
  if (!TryGetIntValue(inputs, kIndex1, kernel_name_, &attr_axis)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get axis input! ";
  }
  if (AnfAlgo::IsDynamicShapeSkipExecute(skip_mode_, inputs[kIndex1]->GetShapeVector())) {
    need_skip_execute_ = true;
    return KRET_OK;
  }

  int input_dim_length = SizeToInt(inputA_shape.size());
  axis_.clear();
  for (auto axis : attr_axis) {
    axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
  }
  std::sort(axis_.begin(), axis_.end());
  auto multiple_pos = std::unique(axis_.begin(), axis_.end());
  axis_.erase(multiple_pos, axis_.end());
  auto outputC_shape = outputs[kIndex0]->GetDeviceShapeAdaptively();
  is_null_input_ =
    CHECK_SHAPE_NULL(inputA_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(outputC_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  InferInAndOutDesc(inputA_shape, outputC_shape);
  if (all_match_) {
    return KRET_OK;
  }
  SimplyReduce(inputA_shape, axis_);
  auto dims = input_reshape_.size();
  if (!(dims == kDim0 || (dims == kDim1 && !reduce_first_axis_) || dims <= kDim3)) {
    workspace_size_list_.push_back(unit_size_ * input_num_);
  }
  return KRET_OK;
}

void ArrayReduceGpuKernelMod::InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape) {
  ShapeVector inputA;
  ShapeVector outputC_shape = output_shape;
  const int split_dim = 4;
  CheckTensorSize({input_shape, output_shape});
  if (input_shape.size() <= split_dim) {
    ShapeNdTo4d(input_shape, &inputA);
  } else {
    inputA = input_shape;
  }

  if (axis_.empty()) {
    outputC_shape.resize(input_shape.size(), 1);

    bool is_not_all_match = std::any_of(inputA.begin(), inputA.end(), [](int64_t s) { return s != 1; });
    if (is_not_all_match) {
      return;
    }

    all_match_ = true;
    return;
  }

  ShapeVector outputC;
  if (!keep_dims_) {
    for (auto i : axis_) {
      (void)(outputC_shape.insert(outputC_shape.begin() + i, 1));
    }
  }

  if (outputC_shape.size() <= split_dim) {
    ShapeNdTo4d(outputC_shape, &outputC);
  } else {
    outputC = output_shape;
  }

  if (inputA == outputC) {
    all_match_ = true;
  }
  return;
}

void ArrayReduceGpuKernelMod::InferArrayReduceType() {
  std::stringstream ss;
  ss << "For '" << kernel_name_ << "', InferArrayReduceType failed.";
  auto iter = kReduceTypeMap.find(kernel_name_);
  if (iter == kReduceTypeMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "Only support these array reduce kernel types: "
                      << "ReduceMax, ReduceMean, ReduceSum, ReduceMin, ReduceAny, ReduceAll, ReduceProd currently"
                      << ", but got " << kernel_name_;
  }
  reduce_op_type_ = iter->second;
  return;
}

std::vector<size_t> ArrayReduceGpuKernelMod::ToRowReduce() {
  const size_t input_reshape_dims = input_reshape_.size();
  std::vector<size_t> shape;
  for (size_t i = reduce_first_axis_; i < input_reshape_dims; i += 2) {
    shape.push_back(input_reshape_[i]);
  }
  for (size_t i = !reduce_first_axis_; i < input_reshape_dims; i += 2) {
    shape.push_back(input_reshape_[i]);
  }
  return shape;
}

void ArrayReduceGpuKernelMod::GetTransposePerm(size_t *transpose_perm) {
  const size_t dims = input_reshape_.size();
  const size_t identity_dims = (dims + !reduce_first_axis_) / 2;
  for (size_t i = 0; i < identity_dims; i++) {
    transpose_perm[i] = 2 * i + reduce_first_axis_;
  }
  for (size_t i = identity_dims; i < dims; i++) {
    transpose_perm[i] = 2 * (i - identity_dims) + !reduce_first_axis_;
  }
  return;
}

void ArrayReduceGpuKernelMod::GetTransposeInfo(TransposeInfo *const info, const size_t dims,
                                               size_t *const transpose_perm) {
  for (size_t i = 0; i < dims; ++i) {
    info->shape[i] = static_cast<int>(input_reshape_[i]);
    info->perm[i] = static_cast<int>(transpose_perm[i]);
  }
  return;
}

std::vector<size_t> ArrayReduceGpuKernelMod::GetNewShape(const size_t dims) {
  input_reshape_ = ToRowReduce();
  size_t new_dim0 = 1;
  size_t new_dim1 = 1;
  std::vector<size_t> new_input_reshape;
  const size_t identity_dims = (dims + !reduce_first_axis_) / 2;
  for (size_t i = 0; i < identity_dims; i++) {
    new_dim0 *= input_reshape_[i];
  }
  new_input_reshape.push_back(new_dim0);
  for (size_t i = identity_dims; i < input_reshape_.size(); i++) {
    new_dim1 *= input_reshape_[i];
  }
  new_input_reshape.push_back(new_dim1);
  return new_input_reshape;
}

template <typename T>
bool ArrayReduceGpuKernelMod::LaunchComplexKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (input_reshape_.size() == kDim0 || (input_reshape_.size() == kDim1 && !reduce_first_axis_)) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, input_addr, input_num_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync output failed");
    return true;
  } else {
    if ((input_reshape_.size() <= kDim3)) {
      auto status = ArrayReduceComplex(input_addr, input_reshape_, reduce_first_axis_, reduce_op_type_, output_addr,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
      return true;
    } else {
      T *temp = GetDeviceAddress<T>(workspace, kIndex0);
      const size_t dims = input_reshape_.size();
      size_t transpose_perm[dims] = {0};
      GetTransposePerm(transpose_perm);
      TransposeInfo info;
      GetTransposeInfo(&info, dims, transpose_perm);
      CalTranspose(input_num_, input_addr, info, input_reshape_.size(), temp,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      auto new_input_reshape = GetNewShape(dims);
      reduce_first_axis_ = false;
      auto status = ArrayReduceComplex(temp, new_input_reshape, reduce_first_axis_, reduce_op_type_, output_addr,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
      return true;
    }
  }
  return true;
}

template <typename T>
bool ArrayReduceGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (input_reshape_.size() == kDim0 ||
      (input_reshape_.size() == kDim1 && !reduce_first_axis_)) {  // to-do, define the output when size == 1
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, input_addr, input_num_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync output failed");
    return true;
  } else {
    if ((input_reshape_.size() <= kDim3)) {
      auto status = ArrayReduce(input_addr, input_reshape_, reduce_first_axis_, reduce_op_type_, output_addr,
                                reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
      return true;
    } else {
      T *temp = GetDeviceAddress<T>(workspace, kIndex0);
      const size_t dims = input_reshape_.size();
      size_t transpose_perm[dims] = {0};
      GetTransposePerm(transpose_perm);
      TransposeInfo info;
      GetTransposeInfo(&info, dims, transpose_perm);
      CalTranspose(input_num_, input_addr, info, input_reshape_.size(), temp,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      auto new_input_reshape = GetNewShape(dims);
      reduce_first_axis_ = false;
      auto status = ArrayReduce(temp, new_input_reshape, reduce_first_axis_, reduce_op_type_, output_addr,
                                reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
      return true;
    }
  }
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceSum,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceSum); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceMin,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceMin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceMax,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceMax); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceAny,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceAny); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceAll,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceAll); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceMean,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceMean); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReduceProd,
                                 []() { return std::make_shared<ArrayReduceGpuKernelMod>(kReduceProd); });
}  // namespace kernel
}  // namespace mindspore
