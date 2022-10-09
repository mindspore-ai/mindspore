/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include <tuple>
#include <set>
#include <numeric>

namespace mindspore {
namespace kernel {
namespace {
void CheckDeviceSm(const KernelAttr &kernel_attr) {
  const int major_sm = GET_MAJOR_SM;
  if (!mindspore::device::gpu::CudaCommon::GetInstance().check_sm() || major_sm >= RECOMMEND_SM) {
    return;
  }

  for (size_t i = 0; i < kernel_attr.GetInputSize(); ++i) {
    if (kernel_attr.GetInputAttr(i).first != kNumberTypeFloat16) {
      continue;
    }

    if (major_sm < MINIUM_SM) {
      MS_LOG(EXCEPTION) << "Half precision ops can be used on Devices which computing capacity is >= " << MINIUM_SM
                        << ", but the current device's computing capacity is " << major_sm;
    }
    MS_LOG(WARNING) << "It is recommended to use devices with a computing capacity >= " << RECOMMEND_SM
                    << ", but the current device's computing capacity is " << major_sm;
    mindspore::device::gpu::CudaCommon::GetInstance().set_check_sm(false);
    return;
  }
}
}  // namespace

int DeprecatedNativeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto cnode = kernel_node_.lock();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "kernel_node_ is not a cnode.";
    return KRET_RESIZE_FAILED;
  }

  MS_LOG(DEBUG) << "Update Args: " << cnode->fullname_with_scope();
  DestroyResource();
  ResetResource();
  if (!Init(cnode)) {
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

void DeprecatedNativeGpuKernelMod::SetGpuRefMapToKernelInfo(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  auto kernel_attrs = GetOpSupport();
  if (kernel_attrs.empty()) {
    return;
  }

  auto index = GetMatchKernelAttrIdxWithException(apply_kernel, kernel_attrs);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const KernelBuildInfo *kernel_build_Info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_Info);
  const auto &matched_kernel_attr = kernel_attrs[index];
  if (!matched_kernel_attr.GetOutInRefMap().empty() || matched_kernel_attr.GetAllOutInRef()) {
    kernel_info->set_ref_map(matched_kernel_attr.GetAllOutInRef(), matched_kernel_attr.GetOutInRefMap());
  }
}

bool NativeGpuKernelMod::GpuCheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr) {
  return kernel::Factory<NativeGpuKernelMod>::Instance().Create(kernel_name)->CheckSupport(kernel_name, kernel_attr);
}

std::vector<KernelAttr> NativeGpuKernelMod::GetAllSupportedList(const std::string &kernel_name) {
  auto iter = support_map_.find(kernel_name);
  if (iter == support_map_.end()) {
    auto kernel_support = GetOpSupport();
    (void)support_map_.emplace(kernel_name, kernel_support);
  }
  return support_map_[kernel_name];
}

bool NativeGpuKernelMod::CheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr_to_check) {
  CheckDeviceSm(kernel_attr_to_check);
  auto kernel_attrs = GetAllSupportedList(kernel_name);
  bool is_match;
  std::tie(is_match, std::ignore) = MatchKernelAttr(kernel_attr_to_check, kernel_attrs);

  if (kernel_attrs[0].GetSkipCheck()) {
    is_match = true;
  }
  return is_match;
}

NativeGpuKernelMod::ReducePrecisonRes NativeGpuKernelMod::ReducePrecisionCheck(const std::string &kernel_name,
                                                                               const KernelAttr &kernel_attr_to_check) {
  std::vector<ReduceDetail> input_reduce_index;
  std::vector<ReduceDetail> output_reduce_index;
  std::vector<KernelAttr> kernel_attr_list = this->GetOpSupport();

  const TypeId from_precision = kNumberTypeInt64;
  const TypeId to_precision = kNumberTypeInt32;
  for (size_t attr_index = 0; attr_index < kernel_attr_list.size(); ++attr_index) {
    auto &cur_kernel_attr = kernel_attr_list[attr_index];
    auto attr_size = cur_kernel_attr.GetInputSize();
    MS_EXCEPTION_IF_ZERO("kernel attr input size", attr_size);
    for (size_t iidx = 0; iidx < kernel_attr_to_check.GetInputSize(); iidx++) {
      auto cur_input_attr = kernel_attr_to_check.GetInputAttr(iidx);
      const auto &type_id = cur_input_attr.first;
      if (type_id == from_precision && cur_kernel_attr.GetInputAttr(iidx % attr_size).first == to_precision) {
        (void)input_reduce_index.emplace_back(iidx, from_precision, to_precision);
        MS_LOG(INFO) << "Kernel [" << kernel_name << "] does not support int64, cast input " << iidx << " to int32.";
      }
    }
    for (size_t oidx = 0; oidx < kernel_attr_to_check.GetOutputSize(); oidx++) {
      auto cur_output_attr = kernel_attr_to_check.GetOutputAttr(oidx);
      const auto &type_id = cur_output_attr.first;
      if (type_id == from_precision && cur_kernel_attr.GetOutputAttr(oidx % attr_size).first == to_precision) {
        (void)output_reduce_index.emplace_back(oidx, from_precision, to_precision);
        MS_LOG(INFO) << "Kernel [" << kernel_name << "] does not support int64, cast output " << oidx << " to int32.";
      }
    }
  }

  if (input_reduce_index.empty() && output_reduce_index.empty()) {
    return std::make_tuple(false, input_reduce_index, output_reduce_index);
  }

  auto reduce_kernel_attr = kernel_attr_to_check;
  const size_t kTwo = 2;
  for (const auto &reduce_item : input_reduce_index) {
    auto reduce_idx = std::get<0>(reduce_item);
    auto cur_attr = reduce_kernel_attr.GetInputAttr(reduce_idx);
    reduce_kernel_attr.SetInputAttr(reduce_idx, std::get<kTwo>(reduce_item), cur_attr.second);
  }
  for (const auto &reduce_item : output_reduce_index) {
    auto reduce_idx = std::get<0>(reduce_item);
    auto cur_attr = reduce_kernel_attr.GetOutputAttr(reduce_idx);
    reduce_kernel_attr.SetOutputAttr(reduce_idx, std::get<kTwo>(reduce_item), cur_attr.second);
  }

  MS_LOG(INFO) << "Kernel [" << kernel_name << "] reduce precision attr: " << reduce_kernel_attr;
  return std::make_tuple(CheckSupport(kernel_name, reduce_kernel_attr), input_reduce_index, output_reduce_index);
}

mindspore::HashMap<std::string, std::vector<KernelAttr>> NativeGpuKernelMod::support_map_{};

std::vector<void *> ConvertPtrs(const std::vector<AddressPtr> &input_ptrs) {
  std::vector<void *> out_ptrs;
  std::transform(input_ptrs.begin(), input_ptrs.end(), std::back_inserter(out_ptrs),
                 [](const auto &cur_addr) { return cur_addr->addr; });
  return out_ptrs;
}

bool ShapeNdTo4d(const ShapeVector &src, ShapeVector *dst) {
  const size_t nd_maximum_size = 4;
  if (src.size() > nd_maximum_size) {
    MS_LOG(ERROR) << src.size() << "-D data is not supported!";
    return false;
  }

  dst->push_back(src.size() < kShapeIndex4th ? 1 : src[src.size() - kShapeIndex4th]);
  dst->push_back(src.size() < kShapeIndex3rd ? 1 : src[src.size() - kShapeIndex3rd]);
  dst->push_back(src.size() < kShapeIndex2nd ? 1 : src[src.size() - kShapeIndex2nd]);
  dst->push_back(src.size() == 0 ? 1 : src[src.size() - kShapeIndex1st]);
  return true;
}

int AxisTransform(const std::string &origin_data_format, const std::string &cal_format, int axis) {
  if (((origin_data_format == kOpFormat_DEFAULT) || (origin_data_format == kOpFormat_NCHW)) &&
      (cal_format == kOpFormat_NHWC)) {
    return kNCHWToNHWCAxisMap[axis];
  } else if (((cal_format == kOpFormat_DEFAULT) || (cal_format == kOpFormat_NCHW)) &&
             (origin_data_format == kOpFormat_NHWC)) {
    return kNHWCToNCHWAxisMap[axis];
  } else {
    return axis;
  }
}

void ShapeNCHW2NHWC(ShapeVector *shape) {
  std::swap((*shape)[kShapeIndex1st], (*shape)[kShapeIndex3rd]);
  std::swap((*shape)[kShapeIndex2nd], (*shape)[kShapeIndex1st]);
}

void ShapeNCDHW2NDHWC(ShapeVector *shape) {
  std::swap((*shape)[kShapeIndex1st], (*shape)[kShapeIndex2nd]);
  std::swap((*shape)[kShapeIndex2nd], (*shape)[kShapeIndex3rd]);
  std::swap((*shape)[kShapeIndex3rd], (*shape)[kShapeIndex4th]);
}

void SetDimA(const ShapeVector &shape, int *dimA, size_t len, const std::string &format) {
  if (shape.size() != len) {
    MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with dimA " << len << "-D.";
  }
  if (Anyone(format, "NCHW", "DefaultFormat", "NCDHW")) {
    for (size_t i = 0; i < len; ++i) {
      dimA[i] = LongToInt(shape[i]);
    }
  } else if (format == "NHWC") {
    dimA[0] = LongToInt(shape[0]);
    dimA[kShapeIndex1st] = LongToInt(shape[kShapeIndex3rd]);
    dimA[kShapeIndex2nd] = LongToInt(shape[kShapeIndex1st]);
    dimA[kShapeIndex3rd] = LongToInt(shape[kShapeIndex2nd]);
  } else {
    MS_LOG(ERROR) << "Unsupported data format " << format;
  }
}

void SetStrideA(const ShapeVector &shape, int *strideA, size_t len, const std::string &format) {
  if (shape.size() != len) {
    MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with strideA " << len << "-D.";
  }
  if (Anyone(format, "NCHW", "DefaultFormat", "NCDHW")) {
    for (size_t i = 0; i < len; ++i) {
      strideA[i] = LongToInt(accumulate(shape.begin() + i + 1, shape.end(), 1, std::multiplies<int64_t>()));
    }
  } else if (format == "NHWC") {
    strideA[0] = LongToInt(shape[kShapeIndex1st] * shape[kShapeIndex2nd] * shape[kShapeIndex3rd]);
    strideA[1] = 1;
    strideA[kShapeIndex2nd] = LongToInt(shape[kShapeIndex2nd] * shape[kShapeIndex3rd]);
    strideA[kShapeIndex3rd] = LongToInt(shape[kShapeIndex3rd]);
  } else {
    MS_LOG(ERROR) << "Unsupported data format " << format;
  }
}

void SetNCHW(const ShapeVector &shape, int *n, int *c, int *h, int *w, const std::string &format) {
  if (Anyone(format, "NCHW", "DefaultFormat")) {
    *n = LongToInt(shape[0]);
    *c = LongToInt(shape[kShapeIndex1st]);
    *h = LongToInt(shape[kShapeIndex2nd]);
    *w = LongToInt(shape[kShapeIndex3rd]);
  } else if (format == "NHWC") {
    *n = LongToInt(shape[0]);
    *c = LongToInt(shape[kShapeIndex3rd]);
    *h = LongToInt(shape[kShapeIndex1st]);
    *w = LongToInt(shape[kShapeIndex2nd]);
  } else {
    MS_LOG(ERROR) << "Unsupported data format " << format;
  }
}

void SetNCDHW(const ShapeVector &shape, int *n, int *c, int *d, int *h, int *w, const std::string &format) {
  if (Anyone(format, "NCDHW", "DefaultFormat")) {
    *n = LongToInt(shape[0]);
    *c = LongToInt(shape[kShapeIndex1st]);
    *d = LongToInt(shape[kShapeIndex2nd]);
    *h = LongToInt(shape[kShapeIndex3rd]);
    *w = LongToInt(shape[kShapeIndex4th]);
  } else if (format == "NDHWC") {
    *n = LongToInt(shape[0]);
    *c = LongToInt(shape[kShapeIndex4th]);
    *d = LongToInt(shape[kShapeIndex1st]);
    *h = LongToInt(shape[kShapeIndex2nd]);
    *w = LongToInt(shape[kShapeIndex3rd]);
  } else {
    MS_LOG(ERROR) << "Unsupported data format " << format;
  }
}

bool CheckBroadcast4TensorOp(const std::vector<int> &A, const std::vector<int> &B, const std::vector<int> &Out) {
  if (A != Out && B != Out) {
    MS_LOG(ERROR) << "Double-sided broadcast was not supported in cudnn of cudnnOpTensor:\n"
                     "InputA must match the corresponding dimension of the destination tensor outC, and each "
                     "dimension of the inputB "
                     "must match the corresponding dimension of outC or must be equal to 1.";
    return false;
  }
  return true;
}

bool CheckTensorSize(const std::initializer_list<ShapeVector> &shapes) {
  for (auto shape : shapes) {
    int64_t total_size = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
    if (total_size >= SHAPE_SIZE_LIMIT) {
      MS_LOG(ERROR) << "The total size of the tensor exceeds the max_limit of 2 Giga-elements, which is " << total_size
                    << " elements (" << shape << ").";
      return false;
    }
  }
  return true;
}

bool CudnnSetTensorNdDescriptor(const ShapeVector &shape, cudnnTensorDescriptor_t descriptor, cudnnDataType_t data_type,
                                const std::string &node_name) {
  if (shape.size() < 3) {
    MS_LOG(ERROR) << "cudnnSetTensorNdDescriptor don't support" << shape.size() << "D.";
    return false;
  }
  const int nbDims = shape.size();
  std::unique_ptr<int[]> dim = std::make_unique<int[]>(nbDims);
  std::unique_ptr<int[]> stride = std::make_unique<int[]>(nbDims);

  for (int i = 0; i < nbDims; i++) {
    dim[i] = LongToInt(shape[i]);
    stride[i] = 1;
  }

  for (int i = nbDims - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * LongToInt(shape[i + 1]);
  }

  cudnnStatus_t status = cudnnSetTensorNdDescriptor(descriptor, data_type, nbDims, dim.get(), stride.get());
  if (status != CUDNN_STATUS_SUCCESS) {
    MS_LOG(ERROR) << "cuDNN Error: cudnnSetTensorNdDescriptor failed | Error Number: " << status << " "
                  << cudnnGetErrorString(status);
    return false;
  }
  return true;
}

bool GetCudnnDataType(const std::string &Type, cudnnDataType_t *out_type) {
  auto type = kCudnnDtypeMap.find(Type);
  if (type == kCudnnDtypeMap.end()) {
    MS_LOG(ERROR) << Type << " is not supported.";
    return false;
  }
  *out_type = type->second;
  return true;
}

bool GetCudaDataType(const std::string &Type, cudaDataType_t *out_type) {
  auto type = kCudaDtypeMap.find(Type);
  if (type == kCudaDtypeMap.end()) {
    MS_LOG(ERROR) << Type << " is not supported.";
    return false;
  }
  *out_type = type->second;
  return true;
}

bool ShapeEqual(const ShapeVector &s1, const ShapeVector &s2) {
  return std::equal(s1.begin(), s1.end(), s2.begin(), s2.end());
}
}  // namespace kernel
}  // namespace mindspore
