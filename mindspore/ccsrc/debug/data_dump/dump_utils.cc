/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/backend/debug/data_dump/dump_utils.h"
#include <dirent.h>
#ifdef ENABLE_DEBUGGER
#include <sys/stat.h>
#endif
#include <map>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>

#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "mindspore/core/utils/file_utils.h"

using mindspore::runtime::DeviceTensorStore;

namespace mindspore {
static std::vector<std::string> g_overflow_operators;

uint32_t ConvertPhysicalDeviceId(uint32_t device_id) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto kernel_runtime = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(device_target, device_id);
  MS_EXCEPTION_IF_NULL(kernel_runtime);
  return kernel_runtime->device_id();
}

std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id, bool is_cst) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  if (is_cst) {
    dump_path += ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/constants/");
  } else {
    dump_path +=
      ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/" + iterator + "/");
  }
  return dump_path;
}

void GetFileKernelName(NotNull<std::string *> kernel_name) {
  const std::string strsrc_to_replace[4] = {"/", "\\", ".", " "};
  const std::string strdst = "_";
  for (const auto &strsrc : strsrc_to_replace) {
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();
    while ((pos = kernel_name->find(strsrc, pos)) != std::string::npos) {
      kernel_name->replace(pos, srclen, strdst);
      pos += dstlen;
    }
  }
}

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> const int_shapes, bool trans_flag) {
  if (trans_flag) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<None>(node)) {
      return;
    }
    *int_shapes = trans::GetRuntimePaddingShape(node, index);
  } else {
    *int_shapes = AnfAlgo::GetOutputDeviceShape(node, index);
  }
}

const DeviceTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                       NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type) {
  const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(node.get());
  if (device_tensors.size() < 1) {
    return nullptr;
  }
  auto device_addr = device_tensors[0];
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool trans_flag = dump_json_parser.trans_flag();
  auto ref_node = device_addr->GetNodeIndex().first;
  MS_EXCEPTION_IF_NULL(ref_node);
  GetDumpIntShape(ref_node, kParameterOutputIndex, int_shapes, trans_flag);
  *host_type = common::AnfAlgo::GetOutputInferDataType(ref_node, kParameterOutputIndex);
  *device_type = AnfAlgo::GetOutputDeviceDataType(ref_node, kParameterOutputIndex);
  return device_addr;
}

void DumpMemToFile(const std::string &file_path, const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                   const TypeId &type, bool trans_flag) {
  auto format = kOpFormat_DEFAULT;
  auto ret = addr.DumpMemToFile(file_path, format, int_shapes, type, trans_flag);
  if (!ret) {
    MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << file_path << ", host_format:" << format
                  << ".!";
  }
}

void DumpToFile(const std::string &file_name, const std::string &dump_str) {
  if (dump_str.empty()) {
    MS_LOG(ERROR) << "Failed to dump empty tensor data.";
    return;
  }

  auto real_path = Common::CreatePrefixPath(file_name);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "CreatePrefixPath failed.";
    return;
  }
  std::string real_path_str = real_path.value();
  ChangeFileMode(real_path_str, S_IWUSR);
  std::ofstream file(real_path_str, std::ofstream::out | std::ofstream::trunc);
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "Open file " << real_path_str << "failed: " << ErrnoToString(errno);
  }
  file << dump_str;
  if (file.bad()) {
    file.close();
    MS_LOG(EXCEPTION) << "Dump string to file " << real_path_str << " failed: " << ErrnoToString(errno);
  }
  file.close();
  ChangeFileMode(real_path_str, S_IRUSR);
}
}  // namespace mindspore
