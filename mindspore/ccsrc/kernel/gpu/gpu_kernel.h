/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNEL_H_

#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include "kernel/kernel.h"
#include "device/gpu/gpu_device_manager.h"
#include "device/gpu/gpu_common.h"
#include "session/anf_runtime_algorithm.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
namespace kernel {
class GpuKernel : public KernelMod {
 public:
  virtual ~GpuKernel() = default;
  virtual bool Init(const CNodePtr &kernel_node) = 0;

 protected:
  virtual void InitResource() {}
  virtual void InitSizeLists() = 0;

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
  // expand Nd Shape to 4d (N in [0,4])
  void ShapeNdTo4d(const std::vector<size_t> &src, std::vector<int> *dst) {
    dst->push_back(src.size() < 4 ? 1 : SizeToInt(src[src.size() - 4]));
    dst->push_back(src.size() < 3 ? 1 : SizeToInt(src[src.size() - 3]));
    dst->push_back(src.size() < 2 ? 1 : SizeToInt(src[src.size() - 2]));
    dst->push_back(src.size() == 0 ? 1 : SizeToInt(src[src.size() - 1]));
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
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNEL_H_
