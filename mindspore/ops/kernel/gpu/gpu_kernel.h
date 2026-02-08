/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/gpu/gpu_kernel_mod.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "plugin/gpu/res_manager/gpu_device_manager.h"
#include "kernel/gpu/gpu_common.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/utils.h"

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

constexpr size_t kHelperDimsNum = 5;

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

static auto Anyone = [](auto &&k, auto &&...args) { return ((args == k) || ...); };

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

std::vector<void *> ConvertPtrs(const std::vector<KernelTensor *> &input_ptrs);

// expand Nd Shape to 4d (N in [0,4])
bool ShapeNdTo4d(const ShapeVector &src, ShapeVector *dst);

template <typename T>
inline T *GetPossiblyNullDeviceAddress(const std::vector<KernelTensor *> &addr_list, size_t index) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return nullptr;
  }
  // Kernels may run normally without workspace, the addr_list[index] maybe nullptr.
  if ((addr_list[index] == nullptr) || (addr_list[index]->size() == 0)) {
    return nullptr;
  }
  if (addr_list[index]->device_ptr() == nullptr) {
    MS_LOG(ERROR) << "The device address is empty, address index:" << index;
    return nullptr;
  }
  return reinterpret_cast<T *>(addr_list[index]->device_ptr());
}
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

// old: format string
void SetDimA(const ShapeVector &shape, int *dimA, size_t len, const std::string &format);

void SetStrideA(const ShapeVector &shape, int *strideA, size_t len, const std::string &format);

void SetNCHW(const ShapeVector &shape, int *n, int *c, int *h, int *w, const std::string &format);

void SetNCDHW(const ShapeVector &shape, int *n, int *c, int *d, int *h, int *w, const std::string &format);

// new: format enum
void SetDimA(const ShapeVector &shape, int *dimA, size_t len, const mindspore::Format &format);

void SetStrideA(const ShapeVector &shape, int *strideA, size_t len, const mindspore::Format &format);

void SetNCHW(const ShapeVector &shape, int *n, int *c, int *h, int *w, const mindspore::Format &format);

void SetNCDHW(const ShapeVector &shape, int *n, int *c, int *d, int *h, int *w, const mindspore::Format &format);

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
T GetDimValue(const std::vector<KernelTensor *> &inputs, const int index, const string kernel_name,
              const TypeId &dim_type) {
  size_t size = abstract::TypeIdSize(dim_type);
  auto dim_gpu_addr =
    std::make_shared<device::DeviceAddress>(inputs[index]->device_ptr(), size, device::DeviceType::kGPU);
  int res = 0;
  device::DeviceContextKey host_key = {device::DeviceType::kGPU, dim_gpu_addr->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  if (dim_type == kNumberTypeInt32) {
    int32_t host_dim = 0;
    host_context->device_res_manager_->Copy(&host_dim, dim_gpu_addr->GetMutablePtr(), size, device::CopyType::kD2H,
                                            dim_gpu_addr->stream_id());
    res = static_cast<T>(host_dim);
  } else if (dim_type == kNumberTypeInt64) {
    int64_t host_dim = 0;
    host_context->device_res_manager_->Copy(&host_dim, dim_gpu_addr->GetMutablePtr(), size, device::CopyType::kD2H,
                                            dim_gpu_addr->stream_id());
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
