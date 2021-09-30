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

#include "debug/debugger/debugger_utils.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "debug/anf_ir_utils.h"
#include "debug/debugger/debugger.h"
#include "runtime/device/gpu/gpu_device_address.h"
#include "debug/data_dump/dump_json_parser.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/kernel.h"

using mindspore::kernel::AddressPtr;
using mindspore::kernel::KernelLaunchInfo;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using KernelGraph = mindspore::session::KernelGraph;
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
static const size_t PARAMETER_OUTPUT_INDEX = 0;

std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size) {
  // define a vector containing real output number
  std::vector<size_t> real_outputs;
  // P.BatchNorm is used for training and inference
  // can add the filter list for more operators here....
  if (node_name == "BatchNorm") {
    MS_LOG(INFO) << "loading node named " << node_name;
    (void)real_outputs.insert(real_outputs.end(), {0, 3, 4});
  } else {
    // by default, TensorLoader will load all outputs
    for (size_t j = 0; j < output_size; ++j) {
      real_outputs.push_back(j);
    }
  }
  return real_outputs;
}

void LoadInputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_) {
  // get inputs
  auto kernel_inputs = launch_info_->inputs_;
  auto input_size = AnfAlgo::GetInputTensorNum(cnode);
  for (size_t j = 0; j < input_size; ++j) {
    auto input_kernel = cnode->input(j + 1);
    std::string input_kernel_name = GetKernelNodeName(input_kernel);
    auto addr = kernel_inputs[j];
    auto type = AnfAlgo::GetOutputInferDataType(input_kernel, PARAMETER_OUTPUT_INDEX);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }
#ifdef ENABLE_GPU
    auto format = kOpFormat_DEFAULT;
    auto gpu_addr = std::make_unique<device::gpu::GPUDeviceAddress>(addr->addr, addr->size, format, type);
    string input_tensor_name = input_kernel_name + ':' + "0";
    ShapeVector int_shapes = trans::GetRuntimePaddingShape(input_kernel, PARAMETER_OUTPUT_INDEX);
    auto ret = gpu_addr->LoadMemToHost(input_tensor_name, exec_order_, format, int_shapes, type, 0, true);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << input_tensor_name << ", host_format:" << format << ".!";
    }
#endif
  }
}

void LoadOutputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_) {
  // get outputs
  auto kernel_outputs = launch_info_->outputs_;
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = AnfAlgo::GetCNodeName(cnode);
  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> real_outputs = CheckRealOutput(node_name, output_size);

  for (size_t j : real_outputs) {
    auto addr = kernel_outputs[j];
    auto type = AnfAlgo::GetOutputInferDataType(cnode, j);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }
#ifdef ENABLE_GPU
    auto format = kOpFormat_DEFAULT;
    auto gpu_addr = std::make_unique<device::gpu::GPUDeviceAddress>(addr->addr, addr->size, format, type);
    string tensor_name = kernel_name + ':' + std::to_string(j);
    ShapeVector int_shapes = trans::GetRuntimePaddingShape(cnode, j);
    auto ret = gpu_addr->LoadMemToHost(tensor_name, exec_order_, format, int_shapes, type, j, false);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
    }
#endif
  }
}

bool CheckReadData(const CNodePtr &cnode) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return false;
  }
  bool read_data = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = debugger->DumpDataEnabledIteration();
  std::string kernel_name = GetKernelNodeName(cnode);
  if (dump_enabled) {
    auto dump_mode = dump_json_parser.dump_mode();
    // dump the node if dump_mode is 0, which means all kernels, or if this kernel is in the kernels list
    if ((dump_mode == 0) || ((dump_mode == 1) && dump_json_parser.NeedDump(kernel_name))) {
      read_data = true;
    }
  } else if (debugger->debugger_enabled()) {
    read_data = debugger->ReadNodeDataRequired(cnode);
  }
  return read_data;
}

void ReadDataAndDump(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return;
  }
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = debugger->DumpDataEnabledIteration();
  if (debugger->debugger_enabled() || dump_json_parser.InputNeedDump()) {
    LoadInputs(cnode, launch_info_, exec_order_);
  }
  if (debugger->debugger_enabled() || dump_json_parser.OutputNeedDump()) {
    LoadOutputs(cnode, launch_info_, exec_order_);
  }
  // Dump kernel
  if (dump_enabled) {
    auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto graph_id = kernel_graph->graph_id();
    debugger->DumpSingleNode(cnode, graph_id);
    // Clear Dumped data when online debugger is not enabled
    if (!debugger->debugger_enabled()) {
      debugger->ClearCurrentData();
    }
  }
  // check if the node is last kernel
  bool last_kernel = !AnfAlgo::IsInplaceNode(cnode, "skip");
  debugger->PostExecuteNode(cnode, last_kernel);
}
}  // namespace mindspore
