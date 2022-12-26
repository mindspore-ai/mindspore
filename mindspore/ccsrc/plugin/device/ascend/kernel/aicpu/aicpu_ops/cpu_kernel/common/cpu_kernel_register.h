/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_CONTEXT_INC_REGISTAR_H_
#define AICPU_CONTEXT_INC_REGISTAR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpu_kernel/inc/cpu_context.h"
#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelRegister {
 public:
  /*
   * get instance.
   * @return CpuKernelRegister &: CpuKernelRegister instance
   */
  static CpuKernelRegister &Instance();

  /*
   * get cpu kernel.
   * param op_type: the op type of kernel
   * @return shared_ptr<CpuKernel>: cpu kernel ptr
   */
  std::shared_ptr<CpuKernel> GetCpuKernel(const std::string &opType);

  /*
   * get all cpu kernel registered op types.
   * @return std::vector<string>: all cpu kernel registered op type
   */
  std::vector<std::string> GetAllRegisteredOpTypes() const;

  /*
   * run cpu kernel.
   * param ctx: context of kernel
   * @return uint32_t: 0->success other->failed
   */
  uint32_t RunCpuKernel(CpuKernelContext &ctx);

  /*
   * run async cpu kernel.
   * @param ctx: context of kernel
   * @param wait_type : event wait type
   * @param wait_id : event wait id
   * @param cb : callback function
   * @return uint32_t: 0->success other->failed
   */
  uint32_t RunCpuKernelAsync(CpuKernelContext &ctx, const uint8_t wait_type, const uint32_t wait_id,
                             std::function<uint32_t()> cb);

  // CpuKernel registration function to register different types of kernel to
  // the factory
  class Registerar {
   public:
    Registerar(const std::string &type, const KERNEL_CREATOR_FUN &fun);
    ~Registerar() = default;

    Registerar(const Registerar &) = delete;
    Registerar(Registerar &&) = delete;
    Registerar &operator=(const Registerar &) = delete;
    Registerar &operator=(Registerar &&) = delete;
  };

 protected:
  CpuKernelRegister() = default;
  ~CpuKernelRegister() = default;

  CpuKernelRegister(const CpuKernelRegister &) = delete;
  CpuKernelRegister(CpuKernelRegister &&) = delete;
  CpuKernelRegister &operator=(const CpuKernelRegister &) = delete;
  CpuKernelRegister &operator=(CpuKernelRegister &&) = delete;

  // register creator, this function will call in the constructor
  void Register(const std::string &type, const KERNEL_CREATOR_FUN &fun);

 private:
  std::map<std::string, KERNEL_CREATOR_FUN> creatorMap_;  // kernel map
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_INC_REGISTAR_H_
