/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of context
 */

#ifndef CPU_KERNELS_CONTEXT_H
#define CPU_KERNELS_CONTEXT_H
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_kernel/inc/cpu_types.h"
#include "cpu_kernel/inc/cpu_tensor.h"
#include "cpu_kernel/inc/cpu_attr_value.h"

namespace aicpu {
class Device;
class NodeDef;
class AICPU_VISIBILITY CpuKernelContext {
  friend class CpuKernelUtils;

 public:
  explicit CpuKernelContext(DeviceType type);
  CpuKernelContext() = delete;
  ~CpuKernelContext() = default;
  CpuKernelContext(const CpuKernelContext &) = delete;
  CpuKernelContext(CpuKernelContext &&) = delete;
  CpuKernelContext &operator=(const CpuKernelContext &) = delete;
  CpuKernelContext &operator=(CpuKernelContext &&) = delete;

  uint32_t Init(NodeDef *nodeDef);

  /*
   * get op type.
   * @return string: op type
   */
  std::string GetOpType() const;

  /*
   * get input tensor.
   * @return Tensor *: not null->success, null->failed
   */
  Tensor *Input(uint32_t index) const;

  /*
   * get output tensor.
   * @return Tensor *: not null->success, null->failed
   */
  Tensor *Output(uint32_t index) const;

  /*
   * get attr.
   * @return AttrValue *: not null->success, null->failed
   */
  AttrValue *GetAttr(std::string name) const;

  /*
   * get input size.
   * @return uint32_t: input size
   */
  uint32_t GetInputsSize() const;

  /*
   * get output size.
   * @return uint32_t: output size
   */
  uint32_t GetOutputsSize() const;

 private:
  std::string op_;                                                      // op type
  std::vector<std::shared_ptr<Tensor> > inputs_;                        // input tensor list
  std::vector<std::shared_ptr<Tensor> > outputs_;                       // out tensor list
  std::unordered_map<std::string, std::shared_ptr<AttrValue> > attrs_;  // attr list
  std::shared_ptr<Device> device_{nullptr};
};
}  // namespace aicpu
#endif  // CPU_KERNELS_CONTEXT_H
