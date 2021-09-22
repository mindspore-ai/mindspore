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
#include "debug/data_dump/dump_utils.h"
#include <map>
#include <vector>
#include <algorithm>

#include "common/trans.h"
#include "utils/ms_context.h"
#include "debug/anf_ir_utils.h"
#include "debug/data_dump/dump_json_parser.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime_manager.h"

namespace mindspore {
uint32_t ConvertPhysicalDeviceId(uint32_t device_id) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto kernel_runtime = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(device_target, device_id);
  MS_EXCEPTION_IF_NULL(kernel_runtime);
  return kernel_runtime->device_id();
}

std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  dump_path += ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/" + iterator);
  return dump_path;
}

void GetFileKernelName(NotNull<std::string *> kernel_name) {
  const std::string strsrc = "/";
  const std::string strdst = "--";
  std::string::size_type pos = 0;
  std::string::size_type srclen = strsrc.size();
  std::string::size_type dstlen = strdst.size();
  while ((pos = kernel_name->find(strsrc, pos)) != std::string::npos) {
    kernel_name->replace(pos, srclen, strdst);
    pos += dstlen;
  }
}

void SetConstNodeId(const AnfNodePtr &node, std::map<std::string, size_t> *const_map) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return;
  }
  std::string node_name = GetKernelNodeName(node);
  MS_EXCEPTION_IF_NULL(const_map);
  auto iter = const_map->find(node_name);
  if (iter == const_map->end()) {
    auto const_idx = const_map->size() + 1;
    (*const_map)[node_name] = const_idx;
  }
}

void GetCNodeConstantId(const CNodePtr &node, std::map<std::string, size_t> *const_map) {
  MS_EXCEPTION_IF_NULL(node);
  auto &inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }
  AnfNodePtr op = inputs[0];

  // CNode/ConstGraph/Const/Parameter
  MS_EXCEPTION_IF_NULL(op);
  if (op->isa<CNode>() || IsValueNode<FuncGraph>(op) || op->isa<Parameter>()) {
    MS_LOG(WARNING) << "Operator must be a primitive.";
  } else {
    // process OP inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
      SetConstNodeId(inputs[i], const_map);
    }
  }
}

void GetConstantId(const session::KernelGraph *graph, std::map<std::string, size_t> *const_map) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> nodes = TopoSort(graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode != graph->get_return()) {
      GetCNodeConstantId(cnode, const_map);
    } else {
      SetConstNodeId(cnode->input(1), const_map);
    }
  }
}

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> int_shapes, bool trans_flag) {
  if (trans_flag) {
    *int_shapes = trans::GetRuntimePaddingShape(node, index);
  } else {
    auto shape = AnfAlgo::GetOutputDeviceShape(node, index);
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(*int_shapes),
                         [](size_t inner_item) { return SizeToInt(inner_item); });
  }
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

uint64_t GetTimeStamp() {
  auto cur_sys_time = std::chrono::system_clock::now();
  uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(cur_sys_time.time_since_epoch()).count();
  return timestamp;
}

std::string GetOpNameWithoutScope(const std::string &fullname_with_scope) {
  const std::string separator("--");
  std::size_t found = fullname_with_scope.rfind(separator);
  std::string op_name;
  if (found != std::string::npos) {
    op_name = fullname_with_scope.substr(found + separator.length());
  }
  return op_name;
}
}  // namespace mindspore
