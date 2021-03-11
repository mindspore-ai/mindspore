/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/tasksink/task_generator.h"

#include <runtime/rt.h>
#include "backend/kernel_compiler/task_stream.h"
#include "utils/ms_utils.h"
#include "runtime/device/ascend/profiling/profiling_utils.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *task_info_list,
                             uint32_t graph_id) {
  MS_LOG(INFO) << "GenTasks start...";
  MS_EXCEPTION_IF_NULL(task_info_list);
  // Traverse graph applykernel list and run
  if (!LaunchAllKernel(anf_node_list, task_info_list, graph_id)) {
    MS_LOG(ERROR) << "LaunchAllKernel failed";
    return false;
  }
  MS_LOG(INFO) << "GenTasks end...";
#ifdef ENABLE_DUMP_IR
  string task_info_name = "task_info_graph." + std::to_string(graph_id);
  mindspore::RDR::RecordTaskDebugInfo(SUBMODULE_ID, task_info_name, task_debug_info_list_);
#endif
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "task_info" + "_graph_" + std::to_string(graph_id) + ".ir";
    DumpTaskInfo(file_path);
  }
  return true;
}

void TaskGenerator::LaunchAddrCleanAkgKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  // akg process
  // set atomic clean addr
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, anf_node_ptr)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicOutputIndexs);
    auto graph = anf_node_ptr->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users = manager->node_users();
    if (node_users[anf_node_ptr].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << anf_node_ptr->ToString() << " is empty.";
    }
    auto depend_node = node_users[anf_node_ptr].pop().first;
    if (!IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
      MS_LOG(EXCEPTION) << "Checking Depend node failed";
    }
    if (node_users[depend_node].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << depend_node->ToString() << " is empty.";
    }
    auto post_node = node_users[depend_node].pop().first;
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(post_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      input->size = device_address->size_;
      kernel_inputs->push_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size: " << clean_output_indexs.size();
  }
}

void TaskGenerator::LaunchAddrCleanKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  if (anf_node_ptr->inputs().size() != 2) {
    LaunchAddrCleanAkgKernel(anf_node_ptr, kernel_inputs);
    return;
  }
  MS_EXCEPTION_IF_NULL(anf_node_ptr->inputs()[1]);
  auto pre_node = (anf_node_ptr->inputs()[1])->cast<CNodePtr>();
  // set clean output addr
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(input->addr);
      input->size = device_address->size_;
      kernel_inputs->push_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexs.size();
  }
  // set clean workspace address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto clean_workspace_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    for (const auto &index : clean_workspace_indexs) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(workspace->addr);
      workspace->size = device_address->size_;
      kernel_inputs->push_back(workspace);
    }
  }
  auto clear_mems = AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicAddMemSize);
  if (kernel_inputs->size() != clear_mems.size()) {
    MS_LOG(EXCEPTION) << "AtomicAddClean kernel inputs size not equal clear memory size,kerenl_inputs size:"
                      << kernel_inputs->size() << ",clean mem size" << clear_mems.size();
  }
}

bool TaskGenerator::LaunchKernel(const CNodePtr &anf_node_ptr, uint32_t stream_id,
                                 std::vector<TaskInfoPtr> *task_info_list) {
  MS_EXCEPTION_IF_NULL(task_info_list);
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  auto kernel_mod = AnfAlgo::GetKernelMod(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_kernel_name(anf_node_ptr->fullname_with_scope());
  auto op_name = AnfAlgo::GetCNodeName(anf_node_ptr);
  if ((op_name == kSplitOpName || op_name == kSplitVOpName) && AnfAlgo::HasNodeAttr(kAttrNonTask, anf_node_ptr)) {
    MS_LOG(INFO) << "Skip task generation for NonTask op " << anf_node_ptr->fullname_with_scope();
    return true;
  }

  if (op_name != kAtomicAddrCleanOpName) {
    size_t input_num = AnfAlgo::GetInputTensorNum(anf_node_ptr);
    for (size_t i = 0; i < input_num; ++i) {
      if (op_name == kDynamicRNNOpName && i == 3) {
        continue;
      }
      if (op_name == kDynamicGRUV2OpName) {
        auto none_index = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(anf_node_ptr, "placeholder_index");
        auto item = find(none_index.begin(), none_index.end(), i);
        if (item != none_index.end()) {
          continue;
        }
      }
      auto real_input_index = AnfAlgo::GetRealInputIndex(anf_node_ptr, i);
      auto device_address = AnfAlgo::GetPrevNodeOutputAddr(anf_node_ptr, real_input_index);
      AddressPtr input = std::make_shared<Address>();
      input->addr = device_address->ptr_;
      input->size = device_address->size_;

      auto prenode_with_index = AnfAlgo::GetPrevNodeOutput(anf_node_ptr, i);
      if (AnfAlgo::IsRealCNodeKernel(prenode_with_index.first)) {
        if ((AnfAlgo::GetCNodeName(prenode_with_index.first) == kSplitOpName ||
             AnfAlgo::GetCNodeName(prenode_with_index.first) == kSplitVOpName) &&
            AnfAlgo::HasNodeAttr(kAttrNonTask, prenode_with_index.first->cast<CNodePtr>())) {
          // use memory offset to implement NonTask Type Split op
          // when op A -> split(NonTask) -> op B, op B's input addr is split's input0's addr + offset
          // offset is split's output index * split's output size
          auto split_input0_device_address = AnfAlgo::GetPrevNodeOutputAddr(prenode_with_index.first, 0);
          input->addr =
            static_cast<uint8_t *>(split_input0_device_address->ptr_) + (prenode_with_index.second * input->size);
          MS_LOG(INFO) << "Change " << anf_node_ptr->fullname_with_scope() << "'s input " << i << " address to "
                       << split_input0_device_address->ptr_ << " + " << prenode_with_index.second * input->size;
        }
      }
      kernel_inputs.push_back(input);
    }

    // No kernel output if output of the cnode is monad, such as LabelSwitch.
    if (!HasAbstractMonad(anf_node_ptr)) {
      size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node_ptr);
      for (size_t i = 0; i < output_num; ++i) {
        auto it = AnfAlgo::GetOutputAddr(anf_node_ptr, i);
        AddressPtr output = std::make_shared<Address>();
        output->addr = it->ptr_;
        output->size = it->size_;
        kernel_outputs.push_back(output);
      }
    }

    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(anf_node_ptr, i);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      workspace->size = device_address->size_;
      kernel_workspaces.push_back(workspace);
    }
  } else {
    LaunchAddrCleanKernel(anf_node_ptr, &kernel_inputs);
  }

  auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
  std::vector<TaskInfoPtr> task_info_ptrs =
    ascend_kernel_mod->GenTask(kernel_inputs, kernel_workspaces, kernel_outputs, stream_id);
  task_info_list->insert(task_info_list->end(), task_info_ptrs.begin(), task_info_ptrs.end());
  auto debug_info = std::make_shared<TaskDebugInfo>();
  debug_info->op_name_ = anf_node_ptr->fullname_with_scope();
  debug_info->task_num_ = task_info_ptrs.size();
  debug_info->stream_id_ = task_info_ptrs[0]->stream_id();
  debug_info->dump_flag_ = task_info_ptrs[0]->dump_flag();
  debug_info->input_addrs_ = kernel_inputs;
  debug_info->output_addrs_ = kernel_outputs;
  debug_info->workspace_addrs_ = kernel_workspaces;
  task_debug_info_list_.push_back(debug_info);
  return true;
}

bool TaskGenerator::LaunchAllKernel(const std::vector<CNodePtr> &anf_node_list,
                                    std::vector<TaskInfoPtr> *task_info_list, uint32_t graph_id) {
  uint32_t current_op_index = 0;
  std::vector<CNodePtr> profiling_cnode_list;
  std::vector<std::string> kernel_name_list;
  for (const auto &anf_node_ptr : anf_node_list) {
    size_t old_size = task_info_list->size();
    uint32_t stream_id = AnfAlgo::GetStreamId(anf_node_ptr);
    MS_EXCEPTION_IF_NULL(anf_node_ptr);
    MS_LOG(INFO) << "Task gen launch begin, current_op_idx:" << current_op_index
                 << " name:" << anf_node_ptr->fullname_with_scope() << ", stream id:" << stream_id;
    if (!LaunchKernel(anf_node_ptr, stream_id, task_info_list)) {
      MS_LOG(ERROR) << "LaunchKernel failed.";
      return false;
    }
    for (size_t i = old_size; i < task_info_list->size(); ++i) {
      profiling_cnode_list.emplace_back(anf_node_ptr);
      kernel_name_list.emplace_back(anf_node_ptr->fullname_with_scope());
    }
    current_op_index++;
  }

  ProfilingUtils::SetGraphKernelName(graph_id, kernel_name_list);
  if (ProfilingManager::GetInstance().IsProfiling()) {
    ProfilingUtils::SetGraphProfilingCNode(graph_id, profiling_cnode_list);
  }

  return true;
}

#ifdef ENABLE_DUMP_IR
void TaskGenerator::DumpTaskInfo(const string &real_filename,
                                 const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  OrderedMap<AnfNodePtr, int32_t> para_map;
  ChangeFileMode(real_filename, S_IRWXU);
  std::ofstream fout(real_filename);

  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_filename << "' failed!";
    return;
  }

  size_t index = 0;
  for (auto &task_debug_info : task_debug_info_list) {
    fout << "op_name:" << task_debug_info->op_name_ << "\n"
         << "task_index:" << index << "\t"
         << "task_num:" << task_debug_info->task_num_ << "\t"
         << "task0_stream_id:" << task_debug_info->stream_id_ << "\t"
         << "task0_type:" << task_debug_info->type_ << "\t"
         << "task0_dump_flag:" << task_debug_info->dump_flag_ << "\n";
    index++;
    if (task_debug_info->input_addrs_.size()) {
      fout << "input address:";
      for (auto &input : task_debug_info->input_addrs_) {
        fout << input->addr << "(" << input->size << ")\t";
      }
      fout << "\n";
    }

    if (task_debug_info->output_addrs_.size()) {
      fout << "output address:";
      for (auto &output : task_debug_info->output_addrs_) {
        fout << output->addr << "(" << output->size << ")\t";
      }
      fout << "\n";
    }

    if (task_debug_info->workspace_addrs_.size()) {
      fout << "workspace address:";
      for (auto &workspace : task_debug_info->workspace_addrs_) {
        fout << workspace->addr << "(" << workspace->size << ")\t";
      }
      fout << "\n";
    }
    fout << "\n";
  }

  fout.close();
  // set file mode to read only by user
  ChangeFileMode(real_filename, S_IRUSR);
}
void TaskGenerator::DumpTaskInfo(const std::string &real_filename) {
  if (real_filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << real_filename << " is too long.";
    return;
  }
  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, filename.c_str(), PATH_MAX) == nullptr) {
    MS_LOG(DEBUG) << "dir " << filename << " does not exit.";
  }
#else
  if (nullptr == realpath(real_filename.c_str(), real_path)) {
    MS_LOG(DEBUG) << "Dir " << real_filename << " does not exit.";
  }
#endif

  OrderedMap<AnfNodePtr, int32_t> para_map;
  std::string path_string = real_path;
  ChangeFileMode(path_string, S_IRWXU);
  std::ofstream fout(real_path);

  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path << "' failed!";
    return;
  }

  size_t index = 0;
  for (auto &task_debug_info : task_debug_info_list_) {
    fout << "op_name:" << task_debug_info->op_name_ << "\n"
         << "task_index:" << index << "\t"
         << "task_num:" << task_debug_info->task_num_ << "\t"
         << "task0_stream_id:" << task_debug_info->stream_id_ << "\t"
         << "task0_type:" << task_debug_info->type_ << "\t"
         << "task0_dump_flag:" << task_debug_info->dump_flag_ << "\n";
    index++;
    if (task_debug_info->input_addrs_.size()) {
      fout << "input address:";
      for (auto &input : task_debug_info->input_addrs_) {
        fout << input->addr << "(" << input->size << ")\t";
      }
      fout << "\n";
    }

    if (task_debug_info->output_addrs_.size()) {
      fout << "output address:";
      for (auto &output : task_debug_info->output_addrs_) {
        fout << output->addr << "(" << output->size << ")\t";
      }
      fout << "\n";
    }

    if (task_debug_info->workspace_addrs_.size()) {
      fout << "workspace address:";
      for (auto &workspace : task_debug_info->workspace_addrs_) {
        fout << workspace->addr << "(" << workspace->size << ")\t";
      }
      fout << "\n";
    }
    fout << "\n";
  }

  fout.close();
  // set file mode to read only by user
  ChangeFileMode(path_string, S_IRUSR);
}
#else
void TaskGenerator::DumpTaskInfo(const std::string &real_filename) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping task debug info is disabled, "
                  << "please enable ENABLE_DUMP_IR with '-D on' and recomiple source.";
}
void TaskGenerator::DumpTaskInfo(const string &real_filename,
                                 const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping task debug info is disabled, "
                  << "please enable ENABLE_DUMP_IR with '-D on' and recomiple source.";
}
#endif
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
