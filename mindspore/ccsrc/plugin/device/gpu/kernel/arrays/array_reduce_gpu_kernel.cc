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

#include <memory>
#include "plugin/device/gpu/kernel/arrays/array_reduce_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "ops/reduce.h"

namespace mindspore {
namespace kernel {
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
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kDynamicAxisInputNum = 2;
constexpr size_t kComplexFloatFlag = 1;
constexpr size_t kComplexDoubleFlag = 2;
constexpr size_t kComplexRate = 2;

const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
  {"ReduceMax", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceMean", CUDNN_REDUCE_TENSOR_AVG},
  {"ReduceSum", CUDNN_REDUCE_TENSOR_ADD},  {"ReduceMin", CUDNN_REDUCE_TENSOR_MIN},
  {"ReduceAny", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceAll", CUDNN_REDUCE_TENSOR_MUL},
  {"ReduceProd", CUDNN_REDUCE_TENSOR_MUL},
};

#define STATIC_REGISTER(INPUTX, T) \
  KernelAttr().AddInputAttr(INPUTX).AddOutputAttr(INPUTX), &ArrayReduceGpuKernelMod::LaunchKernel<T>

#define DYN_REGISTER(INPUTX, AXIS, T) \
  KernelAttr().AddInputAttr(INPUTX).AddInputAttr(AXIS).AddOutputAttr(INPUTX), &ArrayReduceGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::all_any_list_ = {
  {STATIC_REGISTER(kNumberTypeBool, bool)},
  {DYN_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {DYN_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)}};
std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::prod_list_ = {
  {STATIC_REGISTER(kNumberTypeFloat64, double)},
  {STATIC_REGISTER(kNumberTypeFloat32, float)},
  {STATIC_REGISTER(kNumberTypeFloat16, half)},
  {STATIC_REGISTER(kNumberTypeInt8, int8_t)},
  {STATIC_REGISTER(kNumberTypeComplex64, Complex<float>)},
  {STATIC_REGISTER(kNumberTypeComplex128, Complex<double>)},
  {DYN_REGISTER(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {DYN_REGISTER(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};
std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::sum_list_ = {
  {STATIC_REGISTER(kNumberTypeFloat64, double)},
  {STATIC_REGISTER(kNumberTypeFloat32, float)},
  {STATIC_REGISTER(kNumberTypeFloat16, half)},
  {STATIC_REGISTER(kNumberTypeBool, bool)},
  {STATIC_REGISTER(kNumberTypeComplex64, Complex<float>)},
  {STATIC_REGISTER(kNumberTypeComplex128, Complex<double>)},
  {DYN_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {DYN_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};
std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::max_min_mean_list_ = {
  {STATIC_REGISTER(kNumberTypeFloat64, double)},
  {STATIC_REGISTER(kNumberTypeFloat32, float)},
  {STATIC_REGISTER(kNumberTypeFloat16, half)},
  {STATIC_REGISTER(kNumberTypeComplex64, Complex<float>)},
  {STATIC_REGISTER(kNumberTypeComplex128, Complex<double>)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};
std::map<std::string, std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>>>
  ArrayReduceGpuKernelMod::kernel_attr_list_ = {
    {prim::kPrimReduceSum->name(), sum_list_},          {prim::kPrimReduceMean->name(), max_min_mean_list_},
    {prim::kPrimReduceProd->name(), prod_list_},        {prim::kPrimReduceMax->name(), max_min_mean_list_},
    {prim::kPrimReduceMin->name(), max_min_mean_list_}, {prim::kPrimReduceAll->name(), all_any_list_},
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

  auto type_id = kernel_attr.GetInputAttr(kIndex0).first;
  auto type_name = TypeIdLabel(type_id);
  if (type_id == kNumberTypeComplex64) {
    data_type_ = CUDNN_DATA_FLOAT;
    complex_op_type = kComplexFloatFlag;
  } else if (type_id == kNumberTypeComplex128) {
    data_type_ = CUDNN_DATA_DOUBLE;
    complex_op_type = kComplexDoubleFlag;
  } else {
    data_type_ = GetCudnnDataType(type_name);
  }

  InitResource();
  return true;
}

void ArrayReduceGpuKernelMod::InitCudnnResource() {
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                      "cudnnGetTensorSizeInBytes failed.");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                   &workspace_size_),
    "cudnnGetReductionWorkspaceSize failed.");
  workspace_size_list_.push_back(workspace_size_);
  if (complex_op_type != 0) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(outputC_descriptor_, &output_size_),
                                        "cudnnGetTensorSizeInBytes failed.");
    output_size_list_.clear();
    output_size_list_.push_back(output_size_ * kComplexRate);
  }
  return;
}

int ArrayReduceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    InitCudnnResource();
    return ret;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Reduce>(base_operator);
  auto inputA_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();

  std::vector<int64_t> attr_axis;
  if (inputs.size() == kDynamicAxisInputNum) {
    if (AnfAlgo::IsDynamicShapeSkipExecute(kernel_name_, inputs[kIndex1]->GetShapeVector())) {
      need_skip_execute_ = true;
      InitCudnnResource();
      return KRET_OK;
    }
    if (!TryGetIntValue(inputs, kIndex1, kernel_name_, &attr_axis)) {
      InitCudnnResource();
      return KRET_OK;
    }
  } else {
    if (kernel_ptr->HasAttr(kAttrAxis)) {
      attr_axis = kernel_ptr->get_axis();
    }
  }
  keep_dims_ = kernel_ptr->get_keep_dims();

  int input_dim_length = SizeToInt(inputA_shape.size());
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
    InitCudnnResource();
    return KRET_OK;
  }

  InferInAndOutDesc(inputA_shape, outputC_shape);
  InferArrayReduceType();
  InitCudnnResource();
  return KRET_OK;
}

void ArrayReduceGpuKernelMod::InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape) {
  ShapeVector inputA;
  ShapeVector outputC_shape = output_shape;
  const int split_dim = 4;
  CheckTensorSize({input_shape, output_shape});
  if (input_shape.size() <= split_dim) {
    ShapeNdTo4d(input_shape, &inputA);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_, LongToInt(inputA[0]),
                                 LongToInt(inputA[1]), LongToInt(inputA[kIndex2]), LongToInt(inputA[kIndex3])),
      "cudnnSetTensor4dDescriptor failed");
  } else {
    (void)CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_, kernel_name_);
    std::copy(input_shape.begin(), input_shape.end(), std::back_inserter(inputA));
  }

  if (axis_.empty()) {
    outputC_shape.resize(input_shape.size(), 1);
    if (outputC_shape.size() <= split_dim) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, 1, 1, 1, 1),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      (void)CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_name_);
    }

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
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, SizeToInt(outputC[0]),
                                 SizeToInt(outputC[1]), SizeToInt(outputC[kIndex2]), SizeToInt(outputC[kIndex3])),
      "cudnnSetTensor4dDescriptor failed");
  } else {
    (void)CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_name_);
    std::copy(outputC_shape.begin(), outputC_shape.end(), std::back_inserter(outputC));
  }

  if (inputA == outputC) {
    all_match_ = true;
  }
  return;
}

void ArrayReduceGpuKernelMod::InferArrayReduceType() {
  auto iter = kReduceTypeMap.find(kernel_name_);
  if (iter == kReduceTypeMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "Only support these array reduce kernel types: "
                      << "ReduceMax, ReduceMean, ReduceSum, ReduceMin, ReduceAny, ReduceAll, ReduceProd currently"
                      << ", but got " << kernel_name_;
  }
  reduce_tensor_op_ = iter->second;
  // add check for float64
  cudnnDataType_t comp_type = (data_type_ == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, reduce_tensor_op_, comp_type, nan_prop_, reduce_indices_,
                                   CUDNN_32BIT_INDICES),
    "cudnnSetReduceTensorDescriptor failed");
  return;
}

template <typename T, typename S>
void ArrayReduceGpuKernelMod::LaunchComplexKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  S alpha = static_cast<S>(1.0f);
  S beta = static_cast<S>(0.0f);
  T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);
  S *input_real = reinterpret_cast<S *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(input_size_));
  S *input_imag = reinterpret_cast<S *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(input_size_));
  S *output_real = reinterpret_cast<S *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(output_size_));
  S *output_imag = reinterpret_cast<S *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(output_size_));
  if (input_real == nullptr || input_imag == nullptr || output_real == nullptr || output_imag == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory alloc of failed";
  }
  int output_count = output_size_ / sizeof(S);
  int input_count = input_size_ / sizeof(S);
  Real(input_addr, input_real, input_count, reinterpret_cast<cudaStream_t>(stream_ptr));
  Imag(input_addr, input_imag, input_count, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_, &alpha,
                      inputA_descriptor_, input_real, &beta, outputC_descriptor_, output_real),
    "cudnnReduceTensor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_, &alpha,
                      inputA_descriptor_, input_imag, &beta, outputC_descriptor_, output_imag),
    "cudnnReduceTensor failed.");
  ElewiseComplexArith(output_count, BROADCAST_TYPE_COMPLEX, output_real, output_imag, output_addr,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(input_real);
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(input_imag);
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(output_real);
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(output_imag);
  return;
}

template <typename T>
bool ArrayReduceGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  if (all_match_ || need_skip_execute_) {
    MS_LOG(DEBUG)
      << "The corresponding dimensions of the input and output tensors all match. No need to call cuDNN kernel.";
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_addr, input_addr, input_size_, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaMemcpyAsync failed in ArrayReduceGpuKernelMod::Launch.");
    return true;
  }
  if (complex_op_type == kComplexFloatFlag) {
    LaunchComplexKernel<Complex<float>, float>(inputs, workspace, outputs, stream_ptr);
    return true;
  } else if (complex_op_type == kComplexDoubleFlag) {
    LaunchComplexKernel<Complex<double>, double>(inputs, workspace, outputs, stream_ptr);
    return true;
  }
  T alpha = static_cast<T>(1.0f);
  T beta = static_cast<T>(0.0f);
  T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);
  if (data_type_ == CUDNN_DATA_DOUBLE) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_, &alpha,
                        inputA_descriptor_, input_addr, &beta, outputC_descriptor_, output_addr),
      "cudnnReduceTensor failed.");
  } else {
    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_, &alphaf,
                        inputA_descriptor_, input_addr, &betaf, outputC_descriptor_, output_addr),
      "cudnnReduceTensor failed.");
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
