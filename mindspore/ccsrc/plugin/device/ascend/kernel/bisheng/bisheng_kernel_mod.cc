/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/bisheng/bisheng_kernel_mod.h"
#include <libgen.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <fstream>
#include "runtime/kernel.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "include/common/debug/common.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"

namespace mindspore::kernel {
namespace {
constexpr size_t k910ACoreNumber = 32;
static uint64_t kBiShengStartAddr = 0xbadbeef;
static uint64_t kBiShengUniqueName = 0;

std::string GetBishengKernelImplPath(const std::string &binary_name) {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetBishengKernelImplPath), &dl_info) == 0) {
    MS_LOG(EXCEPTION) << "Get dladdr error!";
  }
  std::string cur_so_path = dl_info.dli_fname;
  std::string bisheng_impl_path = std::string(dirname(cur_so_path.data())) + "/ascend/" + binary_name + ".so";
  auto realpath = FileUtils::GetRealPath(bisheng_impl_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Invalid file path, " << bisheng_impl_path << " does not exist.";
  }
  return realpath.value();
}

std::string GetBishengKernelMetaPath(const std::string &binary_name) {
  auto kernel_meta_path = Common::GetKernelMetaTempDir() + binary_name + ".o";
  return kernel_meta_path;
}

std::vector<uint8_t> GeneratorDeviceObject(const std::string &binary_name) {
  std::string impl_host_so_path = GetBishengKernelImplPath(binary_name);
  std::string impl_device_so_path = GetBishengKernelMetaPath(binary_name);
  pid_t pid = fork();
  if (pid == 0) {
    (void)execlp("objcopy", "objcopy", "-O", "binary", "--only-section=__CLANG_OFFLOAD_BUNDLE__sycl-ascend_910",
                 impl_host_so_path.c_str(), impl_device_so_path.c_str(), nullptr);
    _exit(0);
  } else if (pid > 0) {
    int status;
    (void)waitpid(pid, &status, 0);
    if (status != 0) {
      MS_LOG(EXCEPTION) << "Generate device object file failed.";
    }
  }
  auto realpath = FileUtils::GetRealPath(impl_device_so_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Invalid file path, " << impl_device_so_path << " does not exist.";
  }

  // read file to buffer
  std::ifstream ifs(realpath.value());
  if (!ifs.good()) {
    MS_LOG(EXCEPTION) << "File: " << realpath.value() << " does not exist.";
  }

  if (!ifs.is_open()) {
    MS_LOG(EXCEPTION) << "File: " << realpath.value() << " open failed.";
  }

  (void)ifs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(ifs.tellg());
  std::vector<uint8_t> buffer(size, 0);
  (void)ifs.seekg(0, std::ios::beg);
  (void)ifs.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
  ifs.close();
  return buffer;
}
}  // namespace
BiShengKernelMod::BiShengKernelMod() : AscendKernelMod() {}

BiShengKernelMod::~BiShengKernelMod() {
  if (tiling_addr_ != nullptr) {
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    MS_EXCEPTION_IF_NULL(mem_manager);
    mem_manager->FreeMemFromMemPool(tiling_addr_);
  }
}

size_t BiShengKernelMod::BlockDim() { return k910ACoreNumber; }

void BiShengKernelMod::DoTiling(std::vector<void *> *workspace_addrs) {
  MS_EXCEPTION_IF_NULL(workspace_addrs);
  // Get tiling data
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Create bisheng_kernel_args
  BiShengKernelArgs bisheng_args;
  for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(cnode); ++i) {
    (void)bisheng_args.input_shapes.emplace_back(AnfAlgo::GetInputDeviceShape(cnode, i));
  }
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(cnode); ++i) {
    (void)bisheng_args.output_shapes.emplace_back(AnfAlgo::GetOutputDeviceShape(cnode, i));
  }

  std::vector<uint8_t> tiling_data;
  if (GetTilingFunc() == nullptr) {
    MS_LOG(EXCEPTION) << "Node's tiling func must register! Op name: " << cnode->fullname_with_scope();
  }
  auto ret = GetTilingFunc()(bisheng_args, &tiling_data);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Call bisheng tiling failed, ret: " << ret;
  }

  // Malloc tiling_addr
  auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager);
  if (tiling_addr_ != nullptr) {
    mem_manager->FreeMemFromMemPool(tiling_addr_);
  }
  tiling_addr_ = mem_manager->MallocMemFromMemPool(tiling_data.size(), false);
  if (tiling_addr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Call MemoryPool to allocate tiling_addr_ failed. Op name: " << cnode->fullname_with_scope();
  }

  // CopyHostToDevice
  auto rt_ret =
    aclrtMemcpy(tiling_addr_, tiling_data.size(), tiling_data.data(), tiling_data.size(), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api aclrtMemcpy failed, ret: " << rt_ret;
  }

  // Insert to workspace
  (void)workspace_addrs->emplace_back(tiling_addr_);
}

std::vector<TaskInfoPtr> BiShengKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspaces,
                                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  auto device_o = GeneratorDeviceObject(GetBinary());
  rtDevBinary_t binary = {RT_DEV_BINARY_MAGIC_ELF, 0, static_cast<void *>(device_o.data()), device_o.size()};
  void *dev_binary_handle = nullptr;
  auto rt_ret = rtDevBinaryRegister(&binary, &dev_binary_handle);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtDevBinaryRegister failed, error code: " << rt_ret;
  }
  const auto &binary_func = FunctionName();
  rt_ret = rtFunctionRegister(dev_binary_handle, reinterpret_cast<void *>(kBiShengStartAddr), binary_func.c_str(),
                              binary_func.c_str(), 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtFunctionRegister failed, error code: " << rt_ret;
  }
  kBiShengStartAddr += 1;

  std::vector<uint8_t> args;
  std::vector<uint8_t> sm_desc;
  std::vector<uint8_t> meta_data;
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<void *> workspace_addrs;

  // get raw addresses
  GetRawAddress(inputs, &input_data_addrs);
  GetRawAddress(outputs, &output_data_addrs);
  GetRawAddress(workspaces, &workspace_addrs);

  DoTiling(&workspace_addrs);

  stream_id_ = stream_id;
  block_dim_ = BlockDim();
  auto task_info_ptr = std::make_shared<mindspore::ge::model_runner::TbeTaskInfo>(
    GetOpName() + std::to_string(kBiShengUniqueName), stream_id, binary_func, block_dim_, args, 0, sm_desc, nullptr, 0,
    meta_data, input_data_addrs, output_data_addrs, workspace_addrs, NeedDump());
  kBiShengUniqueName += 1;
  return {task_info_ptr};
}
}  // namespace mindspore::kernel
