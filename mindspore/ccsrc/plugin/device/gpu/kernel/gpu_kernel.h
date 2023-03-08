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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_KERNEL_H_

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
#include <tuple>
#include <set>
#include <optional>
#include "kernel/kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_mod.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

// The max_limit of tensor shape size: 2 Giga-elements(2^31, the largest number in 32 bits).
#define SHAPE_SIZE_LIMIT 2147483648

namespace mindspore {
namespace kernel {
constexpr size_t kShapeIndex1st = 1;
constexpr size_t kShapeIndex2nd = 2;
constexpr size_t kShapeIndex3rd = 3;
constexpr size_t kShapeIndex4th = 4;
constexpr size_t kShapeIndex5nd = 5;
constexpr size_t kShapeIndex6rd = 6;
constexpr size_t kShapeIndex7th = 7;

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

// Choose the suitable datatype for cudnn
inline cudnnDataType_t GetCudnnDataType(const std::string &Type) {
  auto type = kCudnnDtypeMap.find(Type);
  if (type == kCudnnDtypeMap.end()) {
    MS_EXCEPTION(TypeError) << Type << " is not supported.";
  }
  return type->second;
}

// Choose the suitable datatype for cublas
inline cudaDataType_t GetCudaDataType(const std::string &Type) {
  auto type = kCudaDtypeMap.find(Type);
  if (type == kCudaDtypeMap.end()) {
    MS_EXCEPTION(TypeError) << Type << " is not supported.";
  }
  return type->second;
}

class NativeGpuKernelMod : public GpuKernelMod {
 public:
  using ReduceDetail = std::tuple<size_t, TypeId, TypeId>;
  using ReducePrecisonRes = std::tuple<bool, std::vector<ReduceDetail>, std::vector<ReduceDetail>>;

  virtual void DestroyResource() noexcept {}
  bool CheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr);
  std::vector<KernelAttr> GetAllSupportedList(const std::string &kernel_name);
  ReducePrecisonRes ReducePrecisionCheck(const std::string &kernel_name, const KernelAttr &kernel_attr);
  static std::vector<KernelAttr> GetGpuSupportedList(const std::string &kernel_name) {
    if (!Factory<NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
      return {};
    }
    return Factory<NativeGpuKernelMod>::Instance().Create(kernel_name)->GetAllSupportedList(kernel_name);
  }
  std::vector<KernelAttr> GetOpSupport() { return {}; }
  static bool GpuCheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr);

  static ReducePrecisonRes GpuReducePrecisionCheck(const std::string &kernel_name, const KernelAttr &kernel_attr) {
    return Factory<NativeGpuKernelMod>::Instance().Create(kernel_name)->ReducePrecisionCheck(kernel_name, kernel_attr);
  }
  enum KernelModType GetKernelModType() const override { return KernelModType::NativeGpuKernelMod; }

 protected:
  virtual void InitResource() {}
  static mindspore::HashMap<std::string, std::vector<KernelAttr>> support_map_;
};

class DeprecatedNativeGpuKernelMod : public NativeGpuKernelMod {
 public:
  virtual ~DeprecatedNativeGpuKernelMod() = default;
  virtual bool Init(const CNodePtr &kernel_node) = 0;

  void SetGpuRefMapToKernelInfo(const CNodePtr &apply_kernel);
  bool IsDynamicShape() { return common::AnfAlgo::IsDynamicShape(kernel_node_.lock()); }
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  enum KernelModType GetKernelModType() const override { return KernelModType::DeprecatedNativeGpuKernelMod; }

 protected:
  std::weak_ptr<CNode> kernel_node_;
  virtual void InitSizeLists() {}
  virtual void InitResource() {}
  virtual void ResetResource() {
    MS_LOG(ERROR) << "kernel must override the `ResetResource()` method when dynamic shape";
  }
  size_t GetMatchKernelAttrIdxWithException(const AnfNodePtr &node, const std::vector<KernelAttr> &kernel_attrs) {
    auto kernel_attr = GetKernelAttrFromNode(node);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, kernel_attrs);
    if (!is_match) {
      MS_LOG(EXCEPTION) << common::AnfAlgo::GetCNodeName(node)
                        << " does not support this kernel data type: " << kernel_attr;
    }
    return index;
  }

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

    if (addr_list[index] == nullptr) {
      auto kernel_node = kernel_node_.lock();
      const std::string &prim_name = (kernel_node == nullptr ? "" : common::AnfAlgo::GetCNodeName(kernel_node));
      MS_LOG(EXCEPTION) << "The device address is nullptr, address index: " << index << ", op name is: " << prim_name;
    }

    if (addr_list[index]->addr == nullptr) {
      auto kernel_node = kernel_node_.lock();
      const std::string &prim_name = (kernel_node == nullptr ? "" : common::AnfAlgo::GetCNodeName(kernel_node));
      MS_LOG(EXCEPTION) << "The memory of device address is nullptr, address index: " << index
                        << ", op name is: " << prim_name;
    }

    if (addr_list[index]->size == 0) {
      auto kernel_node = kernel_node_.lock();
      const std::string &prim_name = (kernel_node == nullptr ? "" : common::AnfAlgo::GetCNodeName(kernel_node));
      MS_LOG(EXCEPTION) << "The size of device address is zero, address index: " << index
                        << ", op name is: " << prim_name;
    }

    return reinterpret_cast<T *>(addr_list[index]->addr);
  }

  std::vector<void *> ConvertPtrs(const std::vector<AddressPtr> &input_ptrs) {
    std::vector<void *> out_ptrs;
    for (auto &cur_addr : input_ptrs) {
      out_ptrs.emplace_back(cur_addr->addr);
    }
    return out_ptrs;
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

  inline std::vector<ValuePtr> GetAttrAndConvertValueTuple(const CNodePtr &kernel_node, const std::string &key) const {
    const PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    const ValuePtr &attr = prim->GetAttr(key);
    if (attr == nullptr) {
      const std::string &prim_name = common::AnfAlgo::GetCNodeName(kernel_node);
      MS_LOG(EXCEPTION) << "The attr(" << key << ") of kernel(" << prim_name << ") not exist";
    }

    auto attr_ptr = attr->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(attr_ptr);

    auto values = attr_ptr->value();
    for (auto value : values) {
      MS_EXCEPTION_IF_NULL(value);
    }
    return values;
  }

  // expand Nd Shape to 4d (N in [0,4])
  void ShapeNdTo4d(const ShapeVector &src, ShapeVector *dst) {
    const size_t nd_maximum_size = 4;
    if (src.size() > nd_maximum_size) {
      MS_EXCEPTION(ValueError) << src.size() << "-D data is not supported!";
    }

    dst->push_back(src.size() < kShapeIndex4th ? 1 : src[src.size() - kShapeIndex4th]);
    dst->push_back(src.size() < kShapeIndex3rd ? 1 : src[src.size() - kShapeIndex3rd]);
    dst->push_back(src.size() < kShapeIndex2nd ? 1 : src[src.size() - kShapeIndex2nd]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - kShapeIndex1st]);
  }

  // expand Nd Shape to 7d (N in [0,7])
  void ShapeNdTo7d(const ShapeVector &src, ShapeVector *dst) {
    const size_t nd_maximum_size = 7;
    if (src.size() > nd_maximum_size) {
      MS_EXCEPTION(ValueError) << src.size() << "-D data is not supported!";
    }

    dst->push_back(src.size() < kShapeIndex7th ? 1 : src[src.size() - kShapeIndex7th]);
    dst->push_back(src.size() < kShapeIndex6rd ? 1 : src[src.size() - kShapeIndex6rd]);
    dst->push_back(src.size() < kShapeIndex5nd ? 1 : src[src.size() - kShapeIndex5nd]);
    dst->push_back(src.size() < kShapeIndex4th ? 1 : src[src.size() - kShapeIndex4th]);
    dst->push_back(src.size() < kShapeIndex3rd ? 1 : src[src.size() - kShapeIndex3rd]);
    dst->push_back(src.size() < kShapeIndex2nd ? 1 : src[src.size() - kShapeIndex2nd]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - kShapeIndex1st]);
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
  inline void CheckTensorSize(const std::initializer_list<ShapeVector> &shapes) {
    for (auto shape : shapes) {
      int64_t total_size = 1;
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
  void CudnnSetTensorNdDescriptor(const ShapeVector &shape, cudnnTensorDescriptor_t descriptor,
                                  cudnnDataType_t data_type, const std::weak_ptr<CNode> &node) {
    if (shape.size() < 3) {
      MS_EXCEPTION(ValueError) << "cudnnSetTensorNdDescriptor don't support" << shape.size() << "D.";
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

  inline bool ShapeEqual(const ShapeVector &s1, const ShapeVector &s2) {
    return std::equal(s1.begin(), s1.end(), s2.begin(), s2.end());
  }
};

std::vector<void *> ConvertPtrs(const std::vector<AddressPtr> &input_ptrs);

// expand Nd Shape to 4d (N in [0,4])
bool ShapeNdTo4d(const ShapeVector &src, ShapeVector *dst);

template <typename T>
inline T *GetPossiblyNullDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return nullptr;
  }
  // Kernels may run normally without workspace, the addr_list[index] maybe nullptr.
  if ((addr_list[index] == nullptr) || (addr_list[index]->size == 0)) {
    return nullptr;
  }
  if (addr_list[index]->addr == nullptr) {
    MS_LOG(ERROR) << "The device address is empty, address index:" << index;
    return nullptr;
  }
  return reinterpret_cast<T *>(addr_list[index]->addr);
}

int AxisTransform(const std::string &origin_data_format, const std::string &cal_format, int axis);

// transpose shape: NCHW To NHWC
void ShapeNCHW2NHWC(ShapeVector *shape);

// transpose shape: NCDHW To NDHWC
void ShapeNCDHW2NDHWC(ShapeVector *shape);

void SetDimA(const ShapeVector &shape, int *dimA, size_t len, const std::string &format);

void SetStrideA(const ShapeVector &shape, int *strideA, size_t len, const std::string &format);

void SetNCHW(const ShapeVector &shape, int *n, int *c, int *h, int *w, const std::string &format);

void SetNCDHW(const ShapeVector &shape, int *n, int *c, int *d, int *h, int *w, const std::string &format);

bool CheckBroadcast4TensorOp(const std::vector<int> &A, const std::vector<int> &B, const std::vector<int> &Out);

// The tensor size is limited to 2G by cudnn.
bool CheckTensorSize(const std::initializer_list<ShapeVector> &shapes);

// set the tensor descriptor for cudnn/cublas
bool CudnnSetTensorNdDescriptor(const ShapeVector &shape, cudnnTensorDescriptor_t descriptor, cudnnDataType_t data_type,
                                const std::string &node_name);

// choose the suitable datatype for cudnn/cublas
bool GetCudnnDataType(const std::string &Type, cudnnDataType_t *out_type);

bool GetCudaDataType(const std::string &Type, cudaDataType_t *out_type);

bool ShapeEqual(const ShapeVector &s1, const ShapeVector &s2);

template <typename T>
T GetDimValue(const std::vector<AddressPtr> &inputs, const int index, const string kernel_name,
              const TypeId &dim_type) {
  size_t size = abstract::TypeIdSize(dim_type);
  auto dim_gpu_addr =
    std::make_shared<device::gpu::GPUDeviceAddress>(inputs[index]->addr, size, kOpFormat_DEFAULT, dim_type);
  int res = 0;
  if (dim_type == kNumberTypeInt32) {
    int32_t host_dim = 0;
    dim_gpu_addr->SyncDeviceToHost(size, &host_dim);
    res = static_cast<T>(host_dim);
  } else if (dim_type == kNumberTypeInt64) {
    int64_t host_dim = 0;
    dim_gpu_addr->SyncDeviceToHost(size, &host_dim);
    res = static_cast<T>(host_dim);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', got unsupported data type of dim: " << dim_type;
  }
  return res;
}
// This is necessary for gpu kernels to support uint8 data type. In cuda, an unsigned,
// 8 bit integral type is represented by an unsigned char, but the MS_REG_GPU_KERNEL
// macros defined below will create compilation errors when datatype T contains a space,
// because the variable created by the macro will also contain a space. So, we solve this
// problem by writing uchar when calling these macros, and expanding uchar after the
// variable has been created.
using uchar = unsigned char;

inline size_t GetTensorSize(std::vector<size_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_GPU_KERNEL_H_
