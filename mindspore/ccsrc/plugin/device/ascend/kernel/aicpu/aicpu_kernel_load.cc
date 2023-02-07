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
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include <dlfcn.h>
#include <unistd.h>
#include <utility>
#include <string>
#include <ios>
#include <fstream>
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/context.h"
#include "include/common/utils/utils.h"
#include "utils/file_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
bool AicpuOpKernelLoad::GetBinaryFileName(const std::string &so_name, const std::string &bin_folder_path,
                                          std::string *bin_file_path) {
  MS_EXCEPTION_IF_NULL(bin_file_path);
  const auto &iter = so_name_and_realpath_map_.find(so_name);
  if (iter != so_name_and_realpath_map_.end()) {
    *bin_file_path = iter->second;
    MS_LOG(INFO) << "so " << so_name << " has bin file path " << bin_file_path;
    return true;
  }

  std::string bin_file_name(bin_folder_path);
  if (bin_file_name.empty()) {
    bin_file_name = "./";
  } else if (bin_file_name.back() != '/') {
    bin_file_name.append("/");
  }

  bin_file_name += so_name;
  auto real_file_path = FileUtils::GetRealPath(bin_file_name.c_str());
  if (!real_file_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << bin_file_name;
    return false;
  }

  auto real_file_path_value = real_file_path.value();
  if (access(real_file_path_value.c_str(), F_OK) == -1) {
    MS_LOG(ERROR) << "Kernel so path:" << real_file_path_value << " is not existed!";
    return false;
  }

  *bin_file_path = real_file_path_value;
  so_name_and_realpath_map_[so_name] = *bin_file_path;
  return true;
}

bool AicpuOpKernelLoad::ReadBytesFromBinaryFile(const std::string &file_name, std::vector<char> *buffer) const {
  std::ifstream file(file_name.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Open file [" << file_name << "] failed";
    return false;
  }

  std::streamsize size = file.tellg();
  if (size <= 0) {
    file.close();
    MS_LOG(ERROR) << "Empty file [" << file_name << "], please check this file.";
    return false;
  }
  if (size > INT_MAX) {
    file.close();
    MS_LOG(ERROR) << "File [" << file_name << "] size [" << size << "] is out of limit[" << INT_MAX << "]";
    return false;
  }

  file.seekg(0, std::ios::beg);
  buffer->resize(size);
  file.read(buffer->data(), size);
  file.close();
  return true;
}

bool AicpuOpKernelLoad::GetSoNeedLoadPath(std::string *file_path) const {
  MS_EXCEPTION_IF_NULL(file_path);
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(const_cast<AicpuOpKernelLoad *>(this)), &dl_info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed!";
    return false;
  }
  std::string cust_kernel_so_path(dl_info.dli_fname);

  auto pos = cust_kernel_so_path.find_last_of('/');
  if (cust_kernel_so_path.empty() || pos == std::string::npos) {
    MS_LOG(ERROR) << "Current path [" << cust_kernel_so_path << "] is invalid.";
    return false;
  }
  auto real_cust_kernel_so_path = cust_kernel_so_path.substr(0, pos);
  if (real_cust_kernel_so_path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "Current path [" << real_cust_kernel_so_path << "] is too long.";
    return false;
  }

  *file_path = real_cust_kernel_so_path;
  return true;
}

bool AicpuOpKernelLoad::PackageBinaryFile(const std::string &so_name,
                                          std::map<std::string, OpKernelBinPtr> *so_name_with_bin_info) {
  std::string bin_folder_path;
  bool ret = GetSoNeedLoadPath(&bin_folder_path);
  if (!ret) {
    MS_LOG(ERROR) << "GetSoNeedLoadPath failed.";
    return false;
  }

  std::string bin_file_path;
  ret = GetBinaryFileName(so_name, bin_folder_path + "/ascend", &bin_file_path);
  if (!ret) {
    MS_LOG(ERROR) << "GetBinaryFileName failed.";
    return false;
  }

  std::vector<char> buffer;
  ret = ReadBytesFromBinaryFile(bin_file_path, &buffer);
  if (!ret) {
    MS_LOG(ERROR) << "ReadBytesFromBinaryFile failed.";
    return false;
  }

  OpKernelBinPtr cust_aicpu_kernel_ptr = std::make_shared<OpKernelBin>(so_name, std::move(buffer));
  if (cust_aicpu_kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Create OpKernelBin object failed.";
    return false;
  }
  so_name_with_bin_info->emplace(so_name, cust_aicpu_kernel_ptr);

  return true;
}

bool AicpuOpKernelLoad::LoadAicpuKernelSo(const AnfNodePtr &node,
                                          const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  std::lock_guard<std::mutex> lock(cust_aicpu_mutex_);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, cnode)) {
    MS_LOG(INFO) << "Current aicpu ops:" << cnode->fullname_with_scope() << " isn't a custom ops.";
    return true;
  }

  std::string so_name = "lib" + common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrCustAicpu) + ".so";
  if (so_name == kLibAicpuKernelSoName || so_name == kLibCpuKernelSoName) {
    MS_LOG(INFO) << "Aicpu so:" << so_name << " is default so.";
    return true;
  }

  kernel_mod_ptr->SetCustSo(so_name);
  rtContext_t rt_cur_ctx = nullptr;
  auto rt_error = rtCtxGetCurrent(&rt_cur_ctx);
  if (rt_error != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret = 0x" << rt_error;
    return false;
  }
  // use current context as resource key
  uintptr_t resource_id = reinterpret_cast<uintptr_t>(rt_cur_ctx);
  auto it = cust_aicpu_so_.find(resource_id);
  if (it != cust_aicpu_so_.end()) {
    auto it_so_name = it->second.find(so_name);
    if (it_so_name != it->second.end()) {
      MS_LOG(INFO) << "Cust aicpu so:" << so_name << " has been loaded.";
      return true;
    }
  }

  std::map<std::string, OpKernelBinPtr> so_name_with_bin_info;
  if (!PackageBinaryFile(so_name, &so_name_with_bin_info)) {
    MS_LOG(ERROR) << "Package binary file failed.";
    return false;
  }

  if (it == cust_aicpu_so_.end()) {
    cust_aicpu_so_[resource_id] = so_name_with_bin_info;
    MS_LOG(INFO) << "Load new aicpu so:" << so_name << "success, resource id:" << resource_id << ".";
    return true;
  }
  auto it_so_name = it->second.find(so_name);
  if (it_so_name == it->second.end()) {
    it->second.insert(so_name_with_bin_info.begin(), so_name_with_bin_info.end());
    MS_LOG(INFO) << "Load cust aicpu so:" << so_name << "success, resource id:" << resource_id << ".";
    return true;
  }
  return true;
}

bool AicpuOpKernelLoad::CacheBinaryFileToDevice(const uintptr_t &resource_id, std::vector<void *> *allocated_mem,
                                                BatchLoadOpFromBufArgs *batch_args) {
  auto it = cust_aicpu_so_.find(resource_id);
  if (it == cust_aicpu_so_.end()) {
    MS_LOG(ERROR) << "Context id:" << resource_id << " is invalid.";
    return false;
  }

  rtError_t status;
  std::vector<CustAicpuSoBuf> v_cust_so;
  for (const auto &it_so : it->second) {
    if (it_so.second->loaded()) {
      continue;
    }
    const auto &so_name = it_so.first;
    const void *aicpu_data = it_so.second->GetBinData();
    uint32_t aicpu_data_length = it_so.second->GetBinDataSize();
    void *d_aicpu_data = nullptr;
    void *d_so_name = nullptr;

    status = rtMalloc(&d_aicpu_data, aicpu_data_length, RT_MEMORY_HBM, 0);
    if (status != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtMalloc failed, size:" << aicpu_data_length << ", ret = 0x" << status;
      return false;
    }
    allocated_mem->emplace_back(d_aicpu_data);

    status = rtMalloc(&d_so_name, so_name.size(), RT_MEMORY_HBM, 0);
    if (status != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtMalloc failed, size:" << so_name.size() << ", ret = 0x" << status;
      return false;
    }
    allocated_mem->emplace_back(d_so_name);

    status = rtMemcpy(d_aicpu_data, aicpu_data_length, aicpu_data, aicpu_data_length, RT_MEMCPY_HOST_TO_DEVICE);
    if (status != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtMemcpy failed, ret = 0x" << status;
      return false;
    }

    status = rtMemcpy(d_so_name, so_name.size(), reinterpret_cast<const void *>(so_name.c_str()), so_name.size(),
                      RT_MEMCPY_HOST_TO_DEVICE);
    if (status != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtMemcpy failed, ret = 0x" << status;
      return false;
    }

    CustAicpuSoBuf cust_aicpu_so_buf;
    cust_aicpu_so_buf.kernelSoBuf = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_aicpu_data));
    cust_aicpu_so_buf.kernelSoBufLen = aicpu_data_length;
    cust_aicpu_so_buf.kernelSoName = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_so_name));
    cust_aicpu_so_buf.kernelSoNameLen = so_name.size();
    v_cust_so.emplace_back(cust_aicpu_so_buf);
    it_so.second->SetLoaded(true);
  }

  if (v_cust_so.empty()) {
    batch_args->soNum = 0;
    return true;
  }

  void *args = nullptr;
  uint32_t args_size = sizeof(CustAicpuSoBuf) * v_cust_so.size();
  status = rtMalloc(&args, args_size, RT_MEMORY_HBM, 0);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMalloc failed, size:" << args_size << ", ret = 0x" << status;
    return false;
  }
  allocated_mem->emplace_back(args);
  status = rtMemcpy(args, args_size, v_cust_so.data(), args_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMemcpy failed, ret = 0x" << status;
    return false;
  }

  batch_args->soNum = v_cust_so.size();
  batch_args->args = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args));
  return true;
}

bool AicpuOpKernelLoad::LaunchAicpuKernelSo() {
  std::lock_guard<std::mutex> lock(cust_aicpu_mutex_);
  if (cust_aicpu_so_.empty()) {
    return true;
  }

  rtContext_t rt_cur_ctx = nullptr;
  rtError_t status = RT_ERROR_NONE;
  status = rtCtxGetCurrent(&rt_cur_ctx);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret = 0x" << status;
    return false;
  }
  // use current context as resource key
  uintptr_t resource_id = reinterpret_cast<uintptr_t>(rt_cur_ctx);
  auto it = cust_aicpu_so_.find(resource_id);
  if (it == cust_aicpu_so_.end()) {
    MS_LOG(INFO) << "Cust aicpu so map is empty, context id:" << resource_id;
    return true;
  }

  std::vector<void *> allocated_mem;
  batch_args_.push_back({});
  auto &batch_args = *batch_args_.rbegin();
  bool ret = CacheBinaryFileToDevice(resource_id, &allocated_mem, &batch_args);
  allocated_mem_list_.emplace_back(std::move(allocated_mem));
  if (!ret) {
    MS_LOG(ERROR) << "CacheBinaryFileToDevice is failed.";
    return false;
  }
  if (batch_args.soNum == 0) {
    MS_LOG(INFO) << "All cust so has been loaded.";
    return true;
  }

  rtStream_t stream = nullptr;
  status = rtStreamCreate(&stream, 0);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtStreamCreate failed, ret = 0x" << status;
    return false;
  }
  stream_list_.emplace_back(stream);
  // launch "batchLoadsoFrombuf" event to device.
  std::string load_event(kBatchLoadBuf);
  status = rtCpuKernelLaunch(nullptr, load_event.c_str(), 1, reinterpret_cast<void *>(&batch_args),
                             sizeof(BatchLoadOpFromBufArgs), nullptr, stream);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtCpuKernelLaunch failed, ret = 0x" << status;
    return false;
  }
  status = rtStreamSynchronize(stream);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtStreamSynchronize failed, ret = 0x" << status;
    return false;
  }

  MS_LOG(INFO) << "Aicpu kernel so launch success.";
  return true;
}

void AicpuOpKernelLoad::FreeDeviceMemory() {
  for (auto allocated_mem : allocated_mem_list_) {
    for (auto mem : allocated_mem) {
      if (mem == nullptr) {
        continue;
      }
      auto rt_error = rtFree(mem);
      if (rt_error != RT_ERROR_NONE) {
        MS_LOG(EXCEPTION) << "Call rtFree failed, ret = 0x" << rt_error;
      }
    }
  }
  allocated_mem_list_.clear();

  for (auto stream : stream_list_) {
    if (stream != nullptr) {
      auto rt_error = rtStreamDestroy(stream);
      if (rt_error != RT_ERROR_NONE) {
        MS_LOG(EXCEPTION) << "Call rtStreamDestroy failed, ret = 0x" << rt_error;
      }
    }
  }
  stream_list_.clear();

  so_name_and_realpath_map_.clear();
  cust_aicpu_so_.clear();
}
}  // namespace kernel
}  // namespace mindspore
