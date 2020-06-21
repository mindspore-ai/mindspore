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

#include "kernel/akg/ascend/akg_ascend_kernel_mod.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "nlohmann/json.hpp"
#include "runtime/rt.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::map;
using std::mutex;
using std::string;
using TbeTaskInfoPtr = std::shared_ptr<ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
/**
 * @brief infotable contain func_stub\blockdim\kernel file buffer
 */
AkgKernelMod::AkgKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {}

void AkgKernelMod::SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }

void AkgKernelMod::SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }

void AkgKernelMod::SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }

const std::vector<size_t> &AkgKernelMod::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &AkgKernelMod::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &AkgKernelMod::GetWorkspaceSizeList() const { return workspace_size_list_; }

void DumpData(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const char *dump_data = getenv("MS_KERNEL_DUMP_DATA");
  if (dump_data) {
    int idx = 0;
    for (const auto &x : inputs) {
      std::vector<char> buf(x->size);
      if (RT_ERROR_NONE != rtMemcpy(buf.data(), buf.size(), reinterpret_cast<const void *>(x->addr), x->size,
                                    RT_MEMCPY_DEVICE_TO_HOST)) {
        MS_LOG(WARNING) << "Call runtime rtMemcpy error.";
        return;
      }

      std::string file_name("input_");
      file_name += std::to_string(idx);
      std::ofstream file(file_name, std::ios::binary);
      if (file.is_open()) {
        (void)file.write(buf.data(), SizeToLong(buf.size()));
        file.close();
        idx++;
      } else {
        MS_LOG(ERROR) << "Open file failed.";
        return;
      }
    }
    idx = 0;
    for (const auto &x : outputs) {
      std::vector<char> buf(x->size);
      if (RT_ERROR_NONE != rtMemcpy(buf.data(), buf.size(), reinterpret_cast<const void *>(x->addr), x->size,
                                    RT_MEMCPY_DEVICE_TO_HOST)) {
        MS_LOG(WARNING) << "Call runtime rtMemcpy error.";
        return;
      }

      std::string file_name("output_");
      file_name += std::to_string(idx);
      std::ofstream file(file_name, std::ios::binary);
      if (file.is_open()) {
        (void)file.write(buf.data(), SizeToLong(buf.size()));
        file.close();
        idx++;
      } else {
        MS_LOG(ERROR) << "Open file failed.";
        return;
      }
    }
  }
}

bool AkgKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == 0) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr.";
    return false;
  }

  uint32_t block_dim = DEFAULT_BLOCK_DIM;  // default blockdim equal to 1.
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim);
  if (func_stub == 0) {
    MS_LOG(ERROR) << "GenFuncStub failed.";
    return false;
  }

  // pack all addresses into a vector.
  std::vector<void *> runtime_args;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  rtL2Ctrl_t *l2ctrl = nullptr;
  auto stream = reinterpret_cast<rtStream_t *>(stream_ptr);
  if (RT_ERROR_NONE != rtKernelLaunch(reinterpret_cast<void *>(func_stub), block_dim, runtime_args.data(),
                                      SizeToUint(sizeof(void *) * runtime_args.size()), l2ctrl, stream)) {
    MS_LOG(ERROR) << "Call runtime rtKernelLaunch error.";
    return false;
  }

  DumpData(inputs, outputs);

  return true;
}

std::vector<TaskInfoPtr> AkgKernelMod::GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (kernel_pack_ == nullptr) {
    MS_LOG(EXCEPTION) << "kernel pack should not be nullptr.";
  }

  std::vector<uint8_t> args;
  uint32_t args_size = 0;
  std::vector<uint8_t> sm_desc;
  void *binary = nullptr;
  uint32_t binary_size = 0;
  std::vector<uint8_t> meta_data;
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<void *> workspace_addrs;

  // pack all addresses into a vector.
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(input_data_addrs),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  uint32_t block_dim = DEFAULT_BLOCK_DIM;  // default blockdim equal to 1.
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim);
  if (func_stub == 0) {
    MS_LOG(EXCEPTION) << "GenFuncStub failed.";
  }

  std::string stub_func = KernelManager::GetStubFuncName(kernel_pack_);

  MS_LOG(DEBUG) << "The block_dim is:" << block_dim;

  TbeTaskInfoPtr task_info_ptr = make_shared<ge::model_runner::TbeTaskInfo>(
    stream_id, stub_func, block_dim, args, args_size, sm_desc, binary, binary_size, meta_data, input_data_addrs,
    output_data_addrs, workspace_addrs);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
