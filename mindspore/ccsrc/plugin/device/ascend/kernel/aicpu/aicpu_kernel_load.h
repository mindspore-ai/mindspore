/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_LOAD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_LOAD_H_

#include <memory>
#include <string>
#include <map>
#include <vector>
#include "runtime/base.h"
#include "base/base.h"
#include "ir/anf.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
constexpr auto kBatchLoadBuf = "batchLoadsoFrombuf";

#pragma pack(push, 1)
struct CustAicpuSoBuf {
  uint64_t kernelSoBuf;
  uint32_t kernelSoBufLen;
  uint64_t kernelSoName;
  uint32_t kernelSoNameLen;
};

struct BatchLoadOpFromBufArgs {
  uint32_t soNum;
  uint64_t args;
};
#pragma pack(pop)

class AicpuOpKernelLoad {
 public:
  AicpuOpKernelLoad() = default;
  ~AicpuOpKernelLoad() = default;

  static AicpuOpKernelLoad &GetInstance() {
    static AicpuOpKernelLoad instance;
    return instance;
  }

  bool LaunchAicpuKernelSo();
  bool LoadAicpuKernelSo(const AnfNodePtr &node, const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr);
  void FreeDeviceMemory();

 private:
  bool GetBinaryFileName(const std::string &so_name, const std::string &bin_folder_path, std::string *bin_file_path);
  bool ReadBytesFromBinaryFile(const std::string &file_name, std::vector<char> *buffer) const;
  bool GetSoNeedLoadPath(std::string *file_path) const;
  bool PackageBinaryFile(const std::string &so_name, std::map<std::string, OpKernelBinPtr> *so_name_with_bin_info);
  bool CacheBinaryFileToDevice(const uintptr_t &resource_id, std::vector<void *> *allocated_mem,
                               BatchLoadOpFromBufArgs *batch_args);

  std::map<std::string, std::string> so_name_and_realpath_map_;
  std::map<uintptr_t, std::map<std::string, OpKernelBinPtr>> cust_aicpu_so_;
  std::mutex cust_aicpu_mutex_;
  std::vector<rtStream_t> stream_list_;
  std::vector<std::vector<void *>> allocated_mem_list_;
  std::vector<BatchLoadOpFromBufArgs> batch_args_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_LOAD_H_
