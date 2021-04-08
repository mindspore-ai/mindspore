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

std::string GenerateDumpPath(uint32_t *device_id) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  if (device_id == nullptr) {
    dump_path += (net_name + "/iteration_" + iterator);
  } else {
    auto physical_device = ConvertPhysicalDeviceId(*device_id);
    dump_path += (net_name + "/device_" + std::to_string(physical_device) + "/iteration_" + iterator);
  }
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
  if (!node->isa<ValueNode>()) {
    return;
  }
  auto iter = const_map->find(node->fullname_with_scope());
  if (iter == const_map->end()) {
    auto const_idx = const_map->size() + 1;
    (*const_map)[node->fullname_with_scope()] = const_idx;
  }
}

void GetCNodeConstantId(const session::KernelGraph *graph, const CNodePtr &node,
                        std::map<std::string, size_t> *const_map) {
  auto &inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }
  AnfNodePtr op = inputs[0];

  // CNode/ConstGraph/Const/Parameter
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
  std::vector<AnfNodePtr> nodes = TopoSort(graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode != graph->get_return()) {
      GetCNodeConstantId(graph, cnode, const_map);
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

void DumpMemToFile(const std::string &file_path, NotNull<const device::DeviceAddress *> addr,
                   const ShapeVector &int_shapes, const TypeId &type, bool trans_flag) {
  auto format = kOpFormat_DEFAULT;
  auto ret = addr->DumpMemToFile(file_path, format, int_shapes, type, trans_flag);
  if (!ret) {
    MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << file_path << ", host_format:" << format
                  << ".!";
  }
}
}  // namespace mindspore
