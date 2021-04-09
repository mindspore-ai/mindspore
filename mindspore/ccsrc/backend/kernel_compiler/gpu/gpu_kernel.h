/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNEL_H_

#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/gpu/gpu_common.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/executor/dynamic_kernel.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
namespace kernel {
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

class GpuDynamicKernel : public device::DynamicKernel {
 public:
  explicit GpuDynamicKernel(const CNodePtr &cnode_ptr) : DynamicKernel(nullptr, cnode_ptr) {}
  ~GpuDynamicKernel() = default;

  void UpdateArgs() override;
  void PostExecute() final { MS_LOG(EXCEPTION) << "`PostExecute()` should not invoked with gpu backend"; };
  void Execute() final { MS_LOG(EXCEPTION) << "`Execute()` should not invoked with gpu backend"; }
};

class GpuKernel : public KernelMod {
 public:
  virtual ~GpuKernel() = default;
  virtual bool Init(const CNodePtr &kernel_node) = 0;
  virtual void ResetResource() noexcept {
    MS_LOG(ERROR) << "kernel must override the `ResetResource()` method when dynamic shape";
  }
  virtual void DestroyResource() noexcept {}
  virtual void PostExecute() {}

  void InitDynamicKernel(const CNodePtr &cnode_ptr) { dynamic_kernel_ = std::make_shared<GpuDynamicKernel>(cnode_ptr); }
  device::DynamicKernelPtr DynamicKernel() const { return dynamic_kernel_; }

 protected:
  virtual void InitResource() {}
  virtual void InitSizeLists() = 0;
  std::weak_ptr<CNode> kernel_node_;

  template <typename T>
  inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
    if (index >= addr_list.size()) {
      MS_LOG(EXCEPTION) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    }
    // Kernels may run normally without workspace, the addr_list[index] maybe nullptr.
    if ((addr_list[index] == nullptr) || (addr_list[index]->size == 0)) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(addr_list[index]->addr);
    return reinterpret_cast<T *>(addr_list[index]->addr);
  }

  template <typename T>
  inline T GetAttr(const CNodePtr &kernel_node, const std::string &key) const {
    const PrimitivePtr &prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    const ValuePtr &attr = prim->GetAttr(key);
    if (attr == nullptr) {
      const std::string &prim_name = AnfAlgo::GetCNodeName(kernel_node);
      MS_LOG(EXCEPTION) << "The attr(" << key << ") of kernel(" << prim_name << ") not exist";
    }
    return GetValue<T>(attr);
  }
  template <typename T>
  inline T GetAttrWithDefault(const CNodePtr &kernel_node, const std::string &key, const T &value) const {
    const PrimitivePtr &prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    const ValuePtr &attr = prim->GetAttr(key);
    if (attr == nullptr) {
      return value;
    }
    return GetValue<T>(attr);
  }
  // expand Nd Shape to 4d (N in [0,4])
  void ShapeNdTo4d(const std::vector<size_t> &src, std::vector<size_t> *dst) {
    if (src.size() > 4) {
      MS_EXCEPTION(ValueError) << src.size() << "-D data is not supported!";
    }
    dst->push_back(src.size() < 4 ? 1 : src[src.size() - 4]);
    dst->push_back(src.size() < 3 ? 1 : src[src.size() - 3]);
    dst->push_back(src.size() < 2 ? 1 : src[src.size() - 2]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - 1]);
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
    std::swap((*shape)[1], (*shape)[3]);
    std::swap((*shape)[2], (*shape)[1]);
  }

  void SetDimA(const std::vector<size_t> &shape, int *dimA, size_t len, const std::string &format) {
    if (shape.size() != len) {
      MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with dimA " << len << "-D.";
    }
    if (format == "NCHW" || format == "DefaultFormat") {
      dimA[0] = SizeToInt(shape[0]);
      dimA[1] = SizeToInt(shape[1]);
      dimA[2] = SizeToInt(shape[2]);
      dimA[3] = SizeToInt(shape[3]);
    } else if (format == "NHWC") {
      dimA[0] = SizeToInt(shape[0]);
      dimA[1] = SizeToInt(shape[3]);
      dimA[2] = SizeToInt(shape[1]);
      dimA[3] = SizeToInt(shape[2]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }
  void SetStrideA(const std::vector<size_t> &shape, int *strideA, size_t len, const std::string &format) {
    if (shape.size() != len) {
      MS_EXCEPTION(ValueError) << "Invalid size of input shape " << shape.size() << "-D with strideA " << len << "-D.";
    }
    if (format == "NCHW" || format == "DefaultFormat") {
      strideA[0] = SizeToInt(shape[1] * shape[2] * shape[3]);
      strideA[1] = SizeToInt(shape[2] * shape[3]);
      strideA[2] = SizeToInt(shape[3]);
      strideA[3] = 1;
    } else if (format == "NHWC") {
      strideA[0] = SizeToInt(shape[1] * shape[2] * shape[3]);
      strideA[1] = 1;
      strideA[2] = SizeToInt(shape[2] * shape[3]);
      strideA[3] = SizeToInt(shape[3]);
    } else {
      MS_LOG(ERROR) << "Unsupported data format " << format;
    }
  }

  void SetNCHW(const std::vector<size_t> &shape, int *n, int *c, int *h, int *w, const std::string &format) {
    if (format == "NCHW" || format == "DefaultFormat") {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[1]);
      *h = SizeToInt(shape[2]);
      *w = SizeToInt(shape[3]);
    } else if (format == "NHWC") {
      *n = SizeToInt(shape[0]);
      *c = SizeToInt(shape[3]);
      *h = SizeToInt(shape[1]);
      *w = SizeToInt(shape[2]);
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

  device::DynamicKernelPtr dynamic_kernel_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNEL_H_
