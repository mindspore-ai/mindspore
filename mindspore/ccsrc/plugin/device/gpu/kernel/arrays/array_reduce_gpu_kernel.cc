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
#include "plugin/device/gpu/kernel/arrays/cast_gpu_kernel.h"

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
constexpr size_t kInt32Flag = 1;
constexpr size_t kInt64Flag = 2;
constexpr size_t kInt16Flag = 3;

const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
  {"ReduceMax", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceMean", CUDNN_REDUCE_TENSOR_AVG},
  {"ReduceSum", CUDNN_REDUCE_TENSOR_ADD},  {"ReduceMin", CUDNN_REDUCE_TENSOR_MIN},
  {"ReduceAny", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceAll", CUDNN_REDUCE_TENSOR_MUL},
  {"ReduceProd", CUDNN_REDUCE_TENSOR_MUL},
};

#define REDUCE_REGISTER(INPUTX, AXIS, T) \
  KernelAttr().AddInputAttr(INPUTX).AddInputAttr(AXIS).AddOutputAttr(INPUTX), &ArrayReduceGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::all_any_list_ = {
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool)},
  {REDUCE_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool)}};
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
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
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
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};
std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::max_min_list_ = {
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
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
};
std::vector<std::pair<KernelAttr, ArrayReduceGpuKernelMod::ReduceFunc>> ArrayReduceGpuKernelMod::mean_list_ = {
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half)},
  {REDUCE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, Complex<float>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, Complex<double>)},
  {REDUCE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, Complex<double>)},
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

  auto type_id = kernel_attr.GetInputAttr(kIndex0).dtype;
  auto type_name = TypeIdLabel(type_id);
  if (type_id == kNumberTypeComplex64) {
    data_type_ = CUDNN_DATA_FLOAT;
    complex_op_type = kComplexFloatFlag;
  } else if (type_id == kNumberTypeComplex128) {
    data_type_ = CUDNN_DATA_DOUBLE;
    complex_op_type = kComplexDoubleFlag;
  } else if (type_id == kNumberTypeInt64) {
    data_type_ = CUDNN_DATA_DOUBLE;
    int_op_type = kInt64Flag;
  } else if (type_id == kNumberTypeInt16) {
    data_type_ = CUDNN_DATA_FLOAT;
    int_op_type = kInt16Flag;
  } else {
    data_type_ = GetCudnnDataType(type_name);
  }
  if (data_type_ == CUDNN_DATA_INT32) {
    data_type_ = CUDNN_DATA_FLOAT;
    int_op_type = kInt32Flag;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Reduce>(base_operator);
  keep_dims_ = kernel_ptr->get_keep_dims();
  skip_mode_ = kernel_ptr->get_skip_mode();

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
  need_skip_execute_ = false;
  all_match_ = false;
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    InitCudnnResource();
    return ret;
  }

  auto inputA_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  std::vector<int64_t> attr_axis;
  if (!TryGetIntValue(inputs, kIndex1, kernel_name_, &attr_axis)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get axis input! ";
  }
  if (AnfAlgo::IsDynamicShapeSkipExecute(skip_mode_, inputs[kIndex1]->GetShapeVector())) {
    need_skip_execute_ = true;
    // As size of input_size_list_ is equal to size of inputs, input_size_list_[0] is safe.
    input_size_ = input_size_list_[0];
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
void ArrayReduceGpuKernelMod::LaunchIntKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  S *input_addr = GetDeviceAddress<S>(inputs, 0);
  S *output_addr = GetDeviceAddress<S>(outputs, 0);

  T alpha = static_cast<T>(1.0f);
  T beta = static_cast<T>(0.0f);

  S *workspace_addr = GetPossiblyNullDeviceAddress<S>(workspace, 0);
  T *casted_input = GetDeviceAddress<T>(inputs, 0);
  T *output_before_cast = GetDeviceAddress<T>(outputs, 0);

  const int input_num = input_size_ / sizeof(T);
  const int output_num = output_size_list_[kIndex0] / sizeof(S);

  Cast(input_num, input_addr, casted_input, reinterpret_cast<cudaStream_t>(stream_ptr), GET_CTX_DEVICE_ID);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_, &alpha,
                      inputA_descriptor_, casted_input, &beta, outputC_descriptor_, output_before_cast),
    "cudnnReduceTensor failed.");
  Cast(output_num, output_before_cast, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr), GET_CTX_DEVICE_ID);
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
  ElewiseComplexArith(output_count, BinaryOpType::kComplex, output_real, output_imag, output_addr,
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
  if (int_op_type == kInt32Flag) {
    LaunchIntKernel<float, int32_t>(inputs, workspace, outputs, stream_ptr);
    return true;
  } else if (int_op_type == kInt64Flag) {
    LaunchIntKernel<double, int64_t>(inputs, workspace, outputs, stream_ptr);
    return true;
  } else if (int_op_type == kInt16Flag) {
    LaunchIntKernel<float, int16_t>(inputs, workspace, outputs, stream_ptr);
    return true;
  }
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
