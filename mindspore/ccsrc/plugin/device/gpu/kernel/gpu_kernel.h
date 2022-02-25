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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_H_

#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <initializer_list>
#include <utility>
#include <map>
#include <memory>
#include <numeric>
#include <functional>
#include <algorithm>
#include "kernel/kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_mod.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/executor/dynamic_kernel.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

// The max_limit of tensor shape size: 2 Giga-elements(2^31, the largest number in 32 bits).
#define SHAPE_SIZE_LIMIT 2147483648

namespace mindspore {
namespace kernel {
constexpr size_t kShapeIndex1st = 1;
constexpr size_t kShapeIndex2nd = 2;
constexpr size_t kShapeIndex3rd = 3;
constexpr size_t kShapeIndex4th = 4;

constexpr size_t kDim2DShapeSize = 4;
constexpr size_t kDim3DShapeSize = 5;
constexpr size_t kPoolingNbDims = kDim3DShapeSize;

static std::map<int, int> kNCHWToNHWCAxisMap = {
  {0, 0},
  {1, 3},
  {2, 1},
  {3, 2},
};
static std::map<int, int> kNHWCToNCHWAxisMap = {
  {0, 0},
  {1, 2},
  {2, 3},
  {3, 1},
};

static auto Anyone = [](auto &&k, auto &&... args) { return ((args == k) || ...); };

inline int CeilDivide(int m, int n) { return (m + n - 1) / n; }

inline int GetPad(int input, int kernel, int stride) {
  return std::max<int>(0, (CeilDivide(input, stride) - 1) * stride + kernel - input);
}

class NativeGpuKernelMod : public GpuKernelMod {
 public:
  virtual ~NativeGpuKernelMod() = default;
  virtual bool Init(const CNodePtr &kernel_node) = 0;
  virtual void ResetResource() noexcept {
    MS_LOG(ERROR) << "kernel must override the `ResetResource()` method when dynamic shape";
  }
  virtual void DestroyResource() noexcept {}

  bool IsDynamicShape() { return common::AnfAlgo::IsDynamicShape(kernel_node_.lock()); }

  void InferOp() override;
  void InitOp() override;

 protected:
  virtual void InitResource() {}
  virtual void InitSizeLists() = 0;
  std::weak_ptr<CNode> kernel_node_;

  inline void ResetSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  template <typename T>
  inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
    if (index >= addr_list.size()) {
      MS_LOG(EXCEPTION) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    }

    if ((addr_list[index] == nullptr) || (addr_list[index]->addr == nullptr) || (addr_list[index]->size == 0)) {
      auto kernel_node = kernel_node_.lock();
      const std::string &prim_name = (kernel_node == nullptr ? "" : common::AnfAlgo::GetCNodeName(kernel_node));
      MS_LOG(EXCEPTION) << "The device address is empty, address index: " << index << ", op name is: " << prim_name;
    }

    return reinterpret_cast<T *>(addr_list[index]->addr);
  }

  template <typename T>
  inline T *GetPossiblyNullDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
    if (index >= addr_list.size()) {
      MS_LOG(EXCEPTION) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    }
    // Kernels may run normally without workspace, the addr_list[index] maybe nullptr.
    if ((addr_list[index] == nullptr) || (addr_list[index]->size == 0)) {
      return nullptr;
    }
    if (addr_list[index]->addr == nullptr) {
      MS_LOG(EXCEPTION) << "The device address is empty, address index:" << index;
    }
    return reinterpret_cast<T *>(addr_list[index]->addr);
  }

  template <typename T>
  inline T GetAttr(const CNodePtr &kernel_node, const std::string &key) const {
    const PrimitivePtr &prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    const ValuePtr &attr = prim->GetAttr(key);
    if (attr == nullptr) {
      const std::string &prim_name = common::AnfAlgo::GetCNodeName(kernel_node);
      MS_LOG(EXCEPTION) << "The attr(" << key << ") of kernel(" << prim_name << ") not exist";
    }
    return GetValue<T>(attr);
  }
  template <typename T>
  inline T GetAttrWithDefault(const CNodePtr &kernel_node, const std::string &key, const T &value) const {
    const PrimitivePtr &prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    const ValuePtr &attr = prim->GetAttr(key);
    if (attr == nullptr) {
      return value;
    }
    return GetValue<T>(attr);
  }
  // expand Nd Shape to 4d (N in [0,4])
  void ShapeNdTo4d(const std::vector<size_t> &src, std::vector<size_t> *dst) {
    const size_t nd_maximum_size = 4;
    if (src.size() > nd_maximum_size) {
      MS_EXCEPTION(ValueError) << src.size() << "-D data is not supported!";
    }

    dst->push_back(src.size() < kShapeIndex4th ? 1 : src[src.size() - kShapeIndex4th]);
    dst->push_back(src.size() < kShapeIndex3rd ? 1 : src[src.size() - kShapeIndex3rd]);
    dst->push_back(src.size() < kShapeIndex2nd ? 1 : src[src.size() - kShapeIndex2nd]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - kShapeIndex1st]);
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

  // transpose shape: NCHW To NHWC
  void ShapeNCHW2NHWC(std::vector<size_t> *shape) {
    std::swap((*shape)[kShapeIndex1st], (*shape)[kShapeIndex3rd]);
    std::swap((*shape)[kShapeIndex2nd], (*shape)[kShapeIndex1st]);
  }

  // transpose shape: NCDHW To NDHWC
  void ShapeNCDHW2NDHWC(std::vector<size_t> *shape) {
    std::swap((*shape)[kShapeIndex1st], (*shape)[kShapeIndex2nd]);
    std::swap((*shape)[kShapeIndex2nd], (*shape)[kShapeIndex3rd]);
    std::swap((*shape)[kShapeIndex3rd], (*shape)[kShapeIndex4th]);
  }

  void SetDimA(const std::vector<size_t> &shape, int *dimA, size_t len, const std::string &format) {
    if (shape.size() != len) {
      MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with dimA " << len << "-D.";
    }
    if (Anyone(format, "NCHW", "DefaultFormat", "NCDHW")) {
      for (size_t i = 0; i < len; ++i) {
        dimA[i] = SizeToInt(shape[i]);
      }
    } else if (format == "NHWC") {
      dimA[0] = SizeToInt(shape[0]);
      dimA[kShapeIndex1st] = SizeToInt(shape[kShapeIndex3rd]);
      dimA[kShapeIndex2nd] = SizeToInt(shape[kShapeIndex1st]);
      dimA[kShapeIndex3rd] = SizeToInt(shape[kShapeIndex2nd]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }
  void SetStrideA(const std::vector<size_t> &shape, int *strideA, size_t len, const std::string &format) {
    if (shape.size() != len) {
      MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with strideA " << len << "-D.";
    }
    if (Anyone(format, "NCHW", "DefaultFormat", "NCDHW")) {
      for (size_t i = 0; i < len; ++i) {
        strideA[i] = SizeToInt(accumulate(shape.begin() + i + 1, shape.end(), 1, std::multiplies<size_t>()));
      }
    } else if (format == "NHWC") {
      strideA[0] = SizeToInt(shape[kShapeIndex1st] * shape[kShapeIndex2nd] * shape[kShapeIndex3rd]);
      strideA[1] = 1;
      strideA[kShapeIndex2nd] = SizeToInt(shape[kShapeIndex2nd] * shape[kShapeIndex3rd]);
      strideA[kShapeIndex3rd] = SizeToInt(shape[kShapeIndex3rd]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }

  void SetNCHW(const std::vector<size_t> &shape, int *n, int *c, int *h, int *w, const std::string &format) {
    if (Anyone(format, "NCHW", "DefaultFormat")) {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[kShapeIndex1st]);
      *h = SizeToInt(shape[kShapeIndex2nd]);
      *w = SizeToInt(shape[kShapeIndex3rd]);
    } else if (format == "NHWC") {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[kShapeIndex3rd]);
      *h = SizeToInt(shape[kShapeIndex1st]);
      *w = SizeToInt(shape[kShapeIndex2nd]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }

  void SetNCDHW(const std::vector<size_t> &shape, int *n, int *c, int *d, int *h, int *w, const std::string &format) {
    if (Anyone(format, "NCDHW", "DefaultFormat")) {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[kShapeIndex1st]);
      *d = SizeToInt(shape[kShapeIndex2nd]);
      *h = SizeToInt(shape[kShapeIndex3rd]);
      *w = SizeToInt(shape[kShapeIndex4th]);
    } else if (format == "NDHWC") {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[kShapeIndex4th]);
      *d = SizeToInt(shape[kShapeIndex1st]);
      *h = SizeToInt(shape[kShapeIndex2nd]);
      *w = SizeToInt(shape[kShapeIndex3rd]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }

  inline void CheckBroadcast4TensorOp(const std::vector<int> &A, const std::vector<int> &B,
                                      const std::vector<int> &Out) {
    if (A != Out && B != Out) {
      MS_EXCEPTION(ValueError)
        << "Double-sided broadcast was not supported in cudnn of cudnnOpTensor:\n"
           "InputA must match the corresponding dimension of the destination tensor outC, and each "
           "dimension of the inputB "
           "must match the corresponding dimension of outC or must be equal to 1.";
    }
  }

  // The tensor size is limited to 2G by cudnn.
  inline void CheckTensorSize(const std::initializer_list<std::vector<size_t>> &shapes) {
    for (auto shape : shapes) {
      size_t total_size = 1;
      for (auto i : shape) {
        total_size *= i;
      }
      if (total_size >= SHAPE_SIZE_LIMIT) {
        MS_EXCEPTION(ValueError) << "The total size of the tensor exceeds the max_limit of 2 Giga-elements, which is "
                                 << total_size << " elements (" << shape << ").";
      }
    }
  }

  // set the tensor descriptor for cudnn/cublas
  void CudnnSetTensorNdDescriptor(const std::vector<size_t> &shape, cudnnTensorDescriptor_t descriptor,
                                  cudnnDataType_t data_type, const std::weak_ptr<CNode> &node) {
    if (shape.size() < 3) {
      MS_EXCEPTION(ValueError) << "cudnnSetTensorNdDescriptor don't support" << shape.size() << "D.";
    }
    const int nbDims = shape.size();
    std::unique_ptr<int[]> dim = std::make_unique<int[]>(nbDims);
    std::unique_ptr<int[]> stride = std::make_unique<int[]>(nbDims);

    for (int i = 0; i < nbDims; i++) {
      dim[i] = SizeToInt(shape[i]);
      stride[i] = 1;
    }

    for (int i = nbDims - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * SizeToInt(shape[i + 1]);
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(node,
                                cudnnSetTensorNdDescriptor(descriptor, data_type, nbDims, dim.get(), stride.get()),
                                "cudnnSetTensorNdDescriptor failed");
  }

  // choose the suitable datatype for cudnn/cublas
  inline cudnnDataType_t GetCudnnDataType(const std::string &Type) {
    auto type = kCudnnDtypeMap.find(Type);
    if (type == kCudnnDtypeMap.end()) {
      MS_EXCEPTION(TypeError) << Type << " is not supported.";
    }
    return type->second;
  }
  inline cudaDataType_t GetCudaDataType(const std::string &Type) {
    auto type = kCudaDtypeMap.find(Type);
    if (type == kCudaDtypeMap.end()) {
      MS_EXCEPTION(TypeError) << Type << " is not supported.";
    }
    return type->second;
  }

  inline std::vector<int64_t> GetTensorIntValue(const tensor::TensorPtr input_tensor, const size_t input_index) {
    std::vector<int64_t> tensor_value;
    MS_EXCEPTION_IF_NULL(input_tensor);
    size_t data_size = input_tensor->DataSize();
    auto tensor_type = input_tensor->Dtype();
    if (tensor_type->type_id() == kNumberTypeInt32) {
      auto tensor_data = reinterpret_cast<int32_t *>(input_tensor->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      tensor_value.assign(tensor_data, tensor_data + data_size);
    } else if (tensor_type->type_id() == kNumberTypeInt64) {
      auto tensor_data = reinterpret_cast<int64_t *>(input_tensor->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      tensor_value.assign(tensor_data, tensor_data + data_size);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the " << input_index
                              << "th input must be a Tensor[Int64] or Tensor[Int32] type, but got "
                              << input_tensor->ToString();
    }
    return tensor_value;
  }

  inline bool ShapeEqual(const std::vector<size_t> &s1, const std::vector<int64_t> &s2) {
    std::vector<size_t> s2_trans;
    std::transform(s2.begin(), s2.end(), std::back_inserter(s2_trans), [](const int64_t &e) { return LongToSize(e); });
    return std::equal(s1.begin(), s1.end(), s2_trans.begin(), s2_trans.end());
  }

  inline bool GetDynamicAttrIntValue(const CNodePtr &kernel_node, const size_t input_index,
                                     std::vector<int64_t> *attr_value) {
    // The value of dynamic attr can only be obtained after the InferShape() is executed
    if (depend_tensor_map_.empty()) {
      MS_LOG(DEBUG) << "For '" << kernel_name_ << "', the depend_tensor_map is currently empty";
      return false;
    }
    auto depend_iter = depend_tensor_map_.find(input_index);
    if (depend_iter == depend_tensor_map_.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to find the " << input_index
                        << "th input in the depend_tensor_map";
    }
    auto input_tensor = depend_iter->second;
    const auto &input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, input_index);
    if (!ShapeEqual(input_shape, input_tensor->shape())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the " << input_index
                        << "th input is different between the InferShape and the TensorShape";
    }
    const auto &data_format = AnfAlgo::GetInputFormat(kernel_node, input_index);
    if (data_format != kOpFormat_DEFAULT && data_format != kOpFormat_NCHW) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "',  the format of the " << input_index
                        << "th input currently should be the default format and does not support " << data_format;
    }
    *attr_value = GetTensorIntValue(input_tensor, input_index);
    return true;
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_H_
