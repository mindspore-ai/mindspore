/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of cpu kernel
 */

#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include "cpu_kernel/inc/cpu_context.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernel {
 public:
  virtual uint32_t Compute(CpuKernelContext &ctx) = 0;

  virtual ~CpuKernel() {}
};

using KERNEL_CREATOR_FUN = std::function<std::shared_ptr<CpuKernel>(void)>;

AICPU_VISIBILITY bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun);

template <typename T, typename... Args>
static inline std::shared_ptr<T> MakeShared(Args &&... args) {
  typedef typename std::remove_const<T>::type T_nc;
  std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}

#define REGISTER_CPU_KERNEL(type, clazz)                 \
  std::shared_ptr<CpuKernel> Creator_##type##_Kernel() { \
    std::shared_ptr<clazz> ptr = nullptr;                \
    ptr = MakeShared<clazz>();                           \
    return ptr;                                          \
  }                                                      \
  bool g_##type##_Kernel_Creator __attribute__((unused)) = RegistCpuKernel(type, Creator_##type##_Kernel)
}  // namespace aicpu
#endif  // CPU_KERNEL_H
