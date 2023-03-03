/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/common/mem_reuse/mem_reuse_checker.h"
#include <fstream>
#include "utils/trace_base.h"

namespace mindspore {
namespace memreuse {
MemReuseChecker &MemReuseChecker::GetInstance() {
  static MemReuseChecker instance{};
  return instance;
}

void MemReuseChecker::CheckSignalOps(const CNodePtr &c_node) const {
  MS_EXCEPTION_IF_NULL(c_node);
  std::string node_name = common::AnfAlgo::GetCNodeName(c_node);
  if (node_name == kSendOpName || node_name == kRecvOpName) {
    MS_LOG(INFO) << "MemReuseChecker check op_name of  Send or Send";
    // get op's info && check
    MS_LOG(INFO) << "op: " << node_name << " in_num: " << common::AnfAlgo::GetInputTensorNum(c_node)
                 << " out_num: " << AnfAlgo::GetOutputTensorNum(c_node);
  }
}

void MemReuseChecker::CheckWorkSpace(const std::vector<size_t> &max_list) {
  for (auto &ma : max_list) {
    total_re_wkspe_size_checker_ += ma;
  }
}

void MemReuseChecker::CheckOutRef(const KernelRefs &kernel_refs, const CNodePtr &c_node, size_t output_idx) const {
  MS_EXCEPTION_IF_NULL(c_node);
  auto key = c_node.get();
  auto iter = kernel_refs.find(key);
  auto node_name = common::AnfAlgo::GetCNodeName(c_node);
  if (iter == kernel_refs.end()) {
    MS_LOG(EXCEPTION) << "kernel [" << node_name << "] has no output tensor, node: " << c_node->DebugString()
                      << " output index: " << output_idx;
  }
  if (output_idx >= iter->second.size()) {
    MS_LOG(INFO) << "invalid cnode: " << c_node->fullname_with_scope().c_str();
    MS_LOG(EXCEPTION) << "The index: " << output_idx
                      << " is out of the size of kernel_output_refs_:" << iter->second.size();
  }
}

int64_t MemReuseChecker::CalculOriInput(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  int64_t static_input_size = 0;
  for (auto &item : graph->inputs()) {
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId ou_type = AnfAlgo::GetOutputDeviceDataType(item, index);
      // parameter has not init by a cnode
      if (ou_type == kTypeUnknown) {
        ou_type = common::AnfAlgo::GetOutputInferDataType(item, index);
      }
      size_t type_size = GetTypeByte(TypeIdToType(ou_type));
      auto shape = AnfAlgo::GetOutputDeviceShape(item, index);
      size_t tensor_size = type_size * SizeOf(shape);
      auto checker_size = SizeToLong(tensor_size);
      static_input_size += checker_size;
    }
  }
  return static_input_size;
}

int64_t MemReuseChecker::CalculOriValue(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  int64_t static_value_size = 0;
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    auto tensor = node_value->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      continue;
    }
    int64_t checker_size = tensor->data().nbytes();
    static_value_size += checker_size;
  }
  return static_value_size;
}

int64_t MemReuseChecker::CalculOriStatic(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  // cal static inputs
  auto static_input_size = CalculOriInput(graph);
  // do not calcul outpput size
  auto statica_value_size = CalculOriValue(graph);
  auto total_ori_static_size = static_input_size + statica_value_size;
  return total_ori_static_size;
}

int64_t MemReuseChecker::CalculOriDy(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  int64_t ori_dy_size = 0;
  auto kerenls = graph->execution_order();
  for (auto &kernel : kerenls) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (auto &dy_size : kernel_mod->GetOutputSizeList()) {
      auto checker_size = SizeToLong(dy_size);
      ori_dy_size += checker_size;
    }
  }
  return ori_dy_size;
}

int64_t MemReuseChecker::CalculOriWk(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  int64_t ori_wk_size = 0;
  auto kerenls = graph->execution_order();
  for (auto &kernel : kerenls) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (auto &wk_size : kernel_mod->GetWorkspaceSizeList()) {
      auto checker_size = SizeToLong(wk_size);
      ori_wk_size += checker_size;
    }
  }
  return ori_wk_size;
}

std::string MemReuseChecker::GetSplitName(const std::string &scope_name) const {
  auto indx = scope_name.rfind(kSplitC);
  if (indx == std::string::npos) {
    return scope_name;
  } else {
    if (indx < scope_name.size() - 1) {
      auto split_name = scope_name.substr(indx + 1);
      return split_name;
    }
    return scope_name;
  }
}

void MemReuseChecker::CheckMemReuseIR(const KernelRefCountPtrList &total_refs_list,
                                      const KernelDefPtrMaps &kernel_def_ptr_list, const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  total_ori_static_size_ = CalculOriStatic(graph);
  total_ori_input_size_ = CalculOriInput(graph);
  total_ori_value_size_ = CalculOriValue(graph);
  total_ori_dy_size_ = CalculOriDy(graph);
  total_ori_wkspace_size_ = CalculOriWk(graph);
  std::string graph_id = std::to_string(graph->graph_id());
  std::string filename = "./memreuse_" + graph_id + ".ir";
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }
  ofs << "all_tensor_refs:\n";
  ofs << "index:"
      << "\tsize:"
      << "\trefcount:"
      << "\ttype:\n";
  for (auto &ref : total_refs_list) {
    ofs << "%" << ref->index_ << "T"
        << "\t"
        << "#" << ref->size_ << "S"
        << "\t" << ref->ref_count_ << "C"
        << "\t" << ref->type_ << "t"
        << "\n";
  }
  ofs << "kernel_def exc_order:\n";
  int def_idx = 0;
  for (auto &def : kernel_def_ptr_list) {
    ExportMemOpIr(def.get(), ofs, def_idx);
    def_idx++;
  }
  ofs.close();
}

void MemReuseChecker::ExportKernelDependence() {
  std::string filename = "./memreuse_dependence.ir";
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }
  size_t i = 0;
  for (const auto &kernel_front : kernel_front_map_) {
    auto kernel = kernel_front.first;
    MS_EXCEPTION_IF_NULL(kernel);
    auto front = kernel_front.second;
    ofs << "[" << i++ << "] " << kernel->scope_full_name() << "\n";
    for (const auto &node : front) {
      MS_EXCEPTION_IF_NULL(node);
      ofs << node->scope_full_name() << "\n";
    }
    ofs << "\n\n";
  }

  ofs.close();
}

bool MemReuseChecker::CheckGraphOutputAssigned(const session::KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  // set real graph output node to be special who's refcount equal kMaxRefCount
  for (const auto &output : graph->outputs()) {
    MS_EXCEPTION_IF_NULL(output);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(output);
    for (size_t i = 0; i < input_num; ++i) {
      if (output->isa<CNode>()) {
        auto cnode = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        auto input_node = cnode->input(i + 1);
        auto kernel_input_with_idx = common::AnfAlgo::VisitKernel(input_node, 0);
        auto kernel_input = kernel_input_with_idx.first;
        MS_EXCEPTION_IF_NULL(kernel_input);
        auto kernel_mod = AnfAlgo::GetKernelMod(kernel_input);
        if (kernel_mod == nullptr) {
          continue;
        }
        auto output_sizes = kernel_mod->GetOutputSizeList();
        if (output_sizes.empty()) {
          continue;
        }
        for (size_t j = 0; j < output_sizes.size(); ++j) {
          if (!AnfAlgo::OutputAddrExist(kernel_input, j)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void MemReuseChecker::ExportMemOpIr(const KernelDef *def, std::ofstream &ofs, int def_idx) const {
  MS_EXCEPTION_IF_NULL(def);
  auto scope_name = def->scope_full_name();
  std::string split_name = GetSplitName(scope_name);
  ofs << "$" << def_idx << "\t" << split_name << "\t" << static_cast<int>(def->type_) << "\t";
  ofs << "inputs[";
  for (auto &in : def->inputs_) {
    for (auto &in_ref : in.second) {
      MS_EXCEPTION_IF_NULL(in_ref);
      ofs << "%" << in_ref->index_ << "T"
          << ",";
    }
  }
  ofs << "]";
  ofs << "\toutpus[";
  for (auto &ou : def->outputs_) {
    for (auto &ou_ref : ou.second) {
      MS_EXCEPTION_IF_NULL(ou_ref);
      ofs << "%" << ou_ref->index_ << "T"
          << ",";
    }
  }
  ofs << "]";
  ofs << "\tstreamID["
      << "@" << def->stream_id() << "]\n";
}

void MemReuseChecker::ExportNormalTensorIR(std::ofstream &ofs) {
  ofs << "all_tensor_refs:\n";
  ofs << "index:"
      << "\tsize:"
      << "\trefcount:\n";
  size_t ou_idx = 0;
  for (auto &ou : nor_output_tensors_) {
    ofs << "%" << ou_idx << "T"
        << "\t"
        << "#" << nor_tensor_sizes_[ou_idx] << "S"
        << "\t";
    auto iter_ref = ptr_refs_.find(ou);
    if (iter_ref != ptr_refs_.end()) {
      ofs << iter_ref->second << "C"
          << "\n";
    } else {
      MS_LOG(EXCEPTION) << "can not find refs for output";
    }
    ou_idx++;
  }
  ofs << "kernel_def exc_order:\n";
}

int MemReuseChecker::GetTensorIdx(const void *in) const {
  auto iter = ptr_idx_.find(in);
  if (iter == ptr_idx_.end()) {
    return kInvalidIndex;
  } else {
    return SizeToInt(iter->second);
  }
}

void MemReuseChecker::ExportNormalOpIr(const std::vector<CNodePtr> &cnodes) {
  std::ofstream ofs("./normal_mem.ir");
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file failed!";
    return;
  }
  ExportNormalTensorIR(ofs);
  size_t node_idx = 0;
  for (const auto &node : cnodes) {
    MS_EXCEPTION_IF_NULL(node);
    ofs << "$" << node_idx << "\t" << GetSplitName(node->fullname_with_scope()) << "\t";
    std::vector<int> in_idx;
    auto iter = node_ins_.find(node.get());
    if (iter != node_ins_.end()) {
      for (auto &in : iter->second) {
        if (GetTensorIdx(in) != kInvalidIndex) {
          in_idx.push_back(GetTensorIdx(in));
        }
      }
    }
    std::vector<int> ou_idx;
    iter = node_ous_.find(node.get());
    if (iter != node_ous_.end()) {
      for (auto &ou : iter->second) {
        if (GetTensorIdx(ou) != kInvalidIndex) {
          ou_idx.push_back(GetTensorIdx(ou));
        }
      }
    }
    ofs << "inputs[";
    for (auto idx : in_idx) {
      bool has_in_ou = std::any_of(ou_idx.begin(), ou_idx.end(), [idx](int odx) { return idx == odx; });
      if (!has_in_ou) {
        ofs << "%" << idx << "T,";
      }
    }
    ofs << "]\toutpus[";
    for (auto odx : ou_idx) {
      ofs << "%" << odx << "T,";
    }
    ofs << "]\tstreamID[@" << AnfAlgo::GetStreamId(node) << "]\n";
    node_idx++;
  }
  ofs.close();
}

void MemReuseChecker::SetTensorFromAndToInfo(const KernelDef *op_def) {
  MS_EXCEPTION_IF_NULL(op_def);
  auto split_name = GetSplitName(op_def->scope_full_name());
  for (auto &in : op_def->inputs_) {
    auto in_tensors = in.second;
    for (auto &tensor : in_tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      auto indx = tensor->index_;
      tensor_to_[indx].push_back(split_name);
    }
  }
  for (auto &ou : op_def->outputs_) {
    auto ou_tensors = ou.second;
    for (auto &tensor : ou_tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      auto indx = tensor->index_;
      tensor_from_[indx].push_back(split_name);
    }
  }
}

void MemReuseChecker::CheckNormalIR(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &cnodes = graph->execution_order();
  for (const auto &node : cnodes) {
    MS_EXCEPTION_IF_NULL(node);
    std::vector<const void *> curr_ous;
    size_t output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      auto it = AnfAlgo::GetOutputAddr(node, i);
      MS_EXCEPTION_IF_NULL(it);
      auto ptr = it->GetPtr();
      nor_output_tensors_.push_back(ptr);
      nor_tensor_sizes_.push_back(it->GetSize());
      curr_ous.push_back(it->GetPtr());
    }
    (void)node_ous_.emplace(node.get(), curr_ous);
    std::vector<const void *> curr_ins;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_num; ++i) {
      if (i + 1 >= node->inputs().size()) {
        MS_LOG(EXCEPTION) << "Input index: " << i
                          << " is larger than input number: " << common::AnfAlgo::GetInputTensorNum(node)
                          << trace::DumpSourceLines(node);
      }
      auto real_input_index = AnfAlgo::GetInputGraphIdxByKernelIdx(node, i);
      auto input = node->input(real_input_index + 1);
      MS_EXCEPTION_IF_NULL(input);
      auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
      if (kernel_with_index.first->isa<Parameter>()) {
        continue;
      }
      auto device_address = AnfAlgo::GetPrevNodeOutputAddr(node, real_input_index);
      MS_EXCEPTION_IF_NULL(device_address);
      nor_input_tensors_.push_back(device_address->GetPtr());
      curr_ins.push_back(device_address->GetPtr());
    }
    (void)node_ins_.emplace(node.get(), curr_ins);
  }
  size_t ou_idx = 0;
  for (const auto &ou : nor_output_tensors_) {
    (void)ptr_idx_.emplace(ou, ou_idx);
    (void)ptr_refs_.emplace(ou, 0);
    ou_idx++;
  }
  for (const auto &in : nor_input_tensors_) {
    if (ptr_idx_.find(in) != ptr_idx_.end()) {
      if (ptr_refs_.find(in) != ptr_refs_.end()) {
        auto iter = ptr_refs_.find(in);
        (iter->second)++;
      } else {
        MS_LOG(EXCEPTION) << "ptr_refs is not equal to ptr_idx";
      }
    }
  }
  ExportNormalOpIr(cnodes);
}

void MemReuseChecker::SetMembuInfos(const KernelDef *op_def, const std::vector<MembufPtr> &membuf_ptr_list) {
  std::vector<MembufPtr> curr_mem_infos;
  for (const auto &mem : membuf_ptr_list) {
    auto mem_checker =
      std::make_shared<Membuf>(mem->status_, mem->size_, mem->offset_, mem->index_, mem->type_, mem->used_kernel_);
    curr_mem_infos.push_back(mem_checker);
  }
  membuf_all_infos_.push_back(curr_mem_infos);
  auto split_name = GetSplitName(op_def->scope_full_name());
  all_split_names_.push_back(split_name);
  SetTensorFromAndToInfo(op_def);
}

void MemReuseChecker::SetAddNewMembuInfos(const KernelDef *op_def, const std::vector<MembufPtr> &membuf_ptr_list,
                                          size_t op_idx) {
  std::vector<MembufPtr> add_new_curr_mem;

  for (const auto &mem : membuf_ptr_list) {
    auto mem_checker =
      std::make_shared<Membuf>(mem->status_, mem->size_, mem->offset_, mem->index_, mem->type_, mem->used_kernel_);
    add_new_curr_mem.push_back(mem_checker);
  }
  add_new_mem_infos_.push_back(add_new_curr_mem);
  auto split_name = GetSplitName(op_def->scope_full_name());
  add_new_names_.push_back(split_name);
  add_new_op_indxs_.push_back(op_idx);
  add_new_stream_ids_.push_back(op_def->stream_id());
}

void MemReuseChecker::ExportEachMembufInfo(std::ofstream &ofs) {
  size_t i = 0;
  std::vector<size_t> each_node_used_size;
  std::vector<size_t> each_node_allocated_size;
  for (const auto &curr_membuf_list : membuf_all_infos_) {
    ofs << all_split_names_.at(i) << "\n";
    ++i;
    ofs << "mem_num\t"
        << "stream_id\t"
        << "status\t"
        << "tensor_idex\t"
        << "mem_size\t"
        << "mem_head\t"
        << "mem_tail\t"
        << "mem_type\t"
        << "used_kernel\n";
    size_t curr_used = 0;
    size_t curr_allocated = 0;
    for (size_t j = 0; j < curr_membuf_list.size(); ++j) {
      auto membuf = curr_membuf_list.at(j);
      auto used_kernel = membuf->used_kernel_->scope_full_name();
      ofs << "&" << j << "\t"
          << "streamID[@" << membuf->used_kernel_->stream_id() << "]"
          << "\t"
          << "#" << static_cast<int>(membuf->status_) << "\t%" << membuf->index_ << "T"
          << "\t" << membuf->size_ << "\t" << membuf->offset_ << "\t\t" << membuf->offset_ + membuf->size_ << "\t"
          << "\t" << static_cast<int>(membuf->type_) << "\t" << GetSplitName(used_kernel) << "\n";
      if (membuf->status_ == kReused) {
        curr_used += membuf->size_;
      }
    }
    if (!curr_membuf_list.empty()) {
      curr_allocated = curr_membuf_list.back()->offset_ + curr_membuf_list.back()->size_;
    }
    each_node_used_size.push_back(curr_used);
    each_node_allocated_size.push_back(curr_allocated);
    ofs << "curr real used size: \t" << curr_used << "\n";
    ofs << "curr allocated size: \t" << curr_allocated << "\n";
    ofs << "\n\n";
  }
  auto optimal_iter = std::max_element(each_node_used_size.begin(), each_node_used_size.end());
  ofs << "theoretical optimal size: " << *optimal_iter << "\n";
  ofs << "each node used size: \n";
  for (auto size : each_node_used_size) {
    ofs << size << "\t";
  }
  ofs << "\n\n";
  ofs << "each node allocated size: \n";
  for (auto size : each_node_allocated_size) {
    ofs << size << "\t";
  }
  ofs << "\n\n";
}

void MemReuseChecker::ExportMembufInfoIR() {
  std::string ir_file_name = "./mem_buf_info.ir";
  std::ofstream ofs(ir_file_name);
  int64_t total_reuse_size = 0;
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << ir_file_name << "] failed!";
  }
  ofs << "Total static size:\t" << total_ori_static_size_ << "\n";
  ofs << "Graph inputs size:\t" << total_ori_input_size_ << "\n";
  ofs << "Value nodes size:\t" << total_ori_value_size_ << "\n";
  ofs << "Total dynamic size:\t" << total_ori_dy_size_ << "\n";
  ofs << "Total workspace size:\t" << total_ori_wkspace_size_ << "\n";
  // get last membuf_list
  if (membuf_all_infos_.empty()) {
    return;
  }
  auto last_membuf_list = membuf_all_infos_.back();
  for (const auto &membuf : last_membuf_list) {
    auto checker_size = SizeToLong(membuf->size_);
    total_reuse_size += checker_size;
  }
  ofs << "After reuse size:\t" << total_reuse_size << "\n\n";
  ExportEachMembufInfo(ofs);
  ofs.close();
}

void MemReuseChecker::ExportAddNewMmebufIR() {
  std::string ir_file_name = "./AddNewMembuf.ir";
  std::ofstream ofs(ir_file_name);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << ir_file_name << "] failed!";
  }
  auto check_idx = add_new_mem_infos_.size();
  if (check_idx == add_new_op_indxs_.size() && check_idx == add_new_names_.size() &&
      check_idx == add_new_stream_ids_.size()) {
    size_t i = 0;
    for (const auto &curr_membuf_list : add_new_mem_infos_) {
      ofs << "op_idx:$" << add_new_op_indxs_.at(i) << "\t" << add_new_names_.at(i) << "\t";
      ofs << "streamID[@" << add_new_stream_ids_.at(i) << "]"
          << "\n";
      i++;
      ofs << "mem_num\t"
          << "status\t"
          << "tensor_idex\t"
          << "mem_size\t"
          << "mem_head\t"
          << "mem_tail\t"
          << "FromOp\t"
          << "ToOp\n";
      for (size_t j = 0; j < curr_membuf_list.size(); ++j) {
        auto membuf = curr_membuf_list.at(j);
        ofs << "&" << j << "\t"
            << "\t"
            << "#" << static_cast<int>(membuf->status_) << "\t%" << membuf->index_ << "T"
            << "\t" << membuf->size_ << "\t" << membuf->offset_ << "\t" << membuf->offset_ + membuf->size_ << "\t";
        auto in_idx_iter = tensor_from_.find(membuf->index_);
        if (in_idx_iter != tensor_from_.end()) {
          for (auto &in_name : in_idx_iter->second) {
            ofs << in_name << ",";
          }
          ofs << "\t";
        }
        auto ou_idx_iter = tensor_to_.find(membuf->index_);
        if (ou_idx_iter != tensor_to_.end()) {
          for (auto &ou_name : ou_idx_iter->second) {
            ofs << ou_name << ",";
          }
          ofs << "\n";
        }
      }
      ofs << "\n";
    }
  }
  ofs.close();
}
}  // namespace memreuse
}  // namespace mindspore
