/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/dvm/kernel_mod_impl/dvm_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <string>
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "tools/profiler/profiling.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
namespace {
std::string ShapesStr(const std::vector<ShapeVector> &shapes) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << "[";
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      if (j != 0) {
        ss << ", ";
      }
      ss << shapes[i][j];
    }
    ss << "]";
  }
  ss << "]";
  return ss.str();
}
}  // namespace

BaseShapePtr DvmInfer::InferShape(const AbstractBasePtrList &args) { return kernel_->InferShape(args); }

std::mutex DvmKernelMod::lock_;

DvmKernelMod::DvmKernelMod(dvm::KernelType kernel_type, const std::string &op_name, const std::string &op_fullname)
    : op_name_(op_name), op_fullname_(op_fullname) {
  kernel_.Reset(kernel_type);
  dump_kernel_ = graphkernel::GraphKernelFlags::GetInstance().dump_as_text;
}

void DvmKernelMod::Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) {
  inputs_type_byte_.clear();
  inputs_type_byte_.reserve(inputs_type.size());
  (void)std::transform(inputs_type.begin(), inputs_type.end(), std::back_inserter(inputs_type_byte_),
                       [](const TypeId &type_id) { return GetTypeByte(TypeIdToType(type_id)); });
  outputs_type_byte_.clear();
  outputs_type_byte_.reserve(outputs_type.size());
  (void)std::transform(outputs_type.begin(), outputs_type.end(), std::back_inserter(outputs_type_byte_),
                       [](const TypeId &type_id) { return GetTypeByte(TypeIdToType(type_id)); });
  input_size_list_.resize(inputs_type.size(), 1);
  output_size_list_.resize(outputs_type.size(), 1);
  inputs_shape_.resize(inputs_type.size());
  outputs_shape_.resize(outputs_type.size());
  inputs_shape_ref_.resize(inputs_type.size());
}

void DvmKernelMod::CodeGen(const std::vector<ShapeVector> &inputs_shape,
                           const std::vector<ShapeVector> &outputs_shape) {
  for (size_t i = 0; i < inputs_shape.size(); ++i) {
    input_size_list_[i] = inputs_type_byte_[i];
    for (auto sh : inputs_shape[i]) {
      input_size_list_[i] *= LongToSize(sh);
    }
    if (input_size_list_[i] == 0 && inputs_shape_ref_[i] != nullptr) {
      skip_launch_ = true;
    }
  }
  for (size_t i = 0; i < outputs_shape.size(); ++i) {
    output_size_list_[i] = outputs_type_byte_[i];
    for (auto sh : outputs_shape[i]) {
      output_size_list_[i] *= LongToSize(sh);
    }
  }
  if (UseSimulationApi()) {
    return;
  }
  size_t workspace_size{0};
  if (!skip_launch_) {
    workspace_size = kernel_.CodeGen();
  }
  if (workspace_size) {
    workspace_size_list_ = {workspace_size};
  }
  if (dump_kernel_) {
    dump_buf_ << "[dvm kernel] " << op_name_ << " " << op_fullname_ << "\n";
    if (skip_launch_) {
      dump_buf_ << "This kernel is skipped\n";
    } else {
      dump_buf_ << kernel_.Dump() << "\n";
      dump_buf_ << kernel_.Das() << "\n";
    }
    DumpToFile();
  }
}

BaseShapePtr DvmKernelMod::InferShape(const AbstractBasePtrList &inputs_abs) {
  skip_launch_ = false;
  BaseShapePtr result{nullptr};
  // update input shape
  for (size_t i = 0; i < inputs_abs.size(); ++i) {
    // to do: if input value is needed in infer shape, then we should fetch input value here
    inputs_shape_[i] = BaseShapeToShape(inputs_abs[i]->GetShape());
    input_size_list_[i] = inputs_type_byte_[i];
    for (auto sh : inputs_shape_[i]) {
      input_size_list_[i] *= LongToSize(sh);
    }
    if (inputs_shape_ref_[i] != nullptr) {
      *inputs_shape_ref_[i] = inputs_shape_[i];
      if (input_size_list_[i] == 0) {
        skip_launch_ = true;
      }
    }
  }
  if (dump_kernel_) {
    dump_buf_ << "[inputs shape list] " << op_name_ << " " << op_fullname_ << "\n";
    dump_buf_ << ShapesStr(inputs_shape_) << "\n";
    DumpToFile();
  }

  // re-codegen by new input shape
  size_t workspace_size{0};
  if (skip_launch_) {
    kernel_.Infer();
  } else {
    workspace_size = kernel_.CodeGen();
  }
  if (workspace_size) {
    workspace_size_list_ = {workspace_size};
  }
  // update output shape
  UpdateOutputShapes();

  if (dump_kernel_) {
    dump_buf_ << "[outputs shape list] " << op_name_ << " " << op_fullname_ << "\n";
    dump_buf_ << ShapesStr(outputs_shape_) << "\n";
    dump_buf_ << "[dvm kernel] " << op_name_ << " " << op_fullname_ << "\n";
    if (skip_launch_) {
      dump_buf_ << "This kernel is skipped\n";
    } else {
      dump_buf_ << kernel_.Dump() << "\n";
      dump_buf_ << kernel_.Das() << "\n";
    }
    DumpToFile();
  }

  // update output abstract
  if (outputs_shape_.size() > 1) {
    abstract::BaseShapePtrList out_shapes(outputs_shape_.size());
    for (size_t i = 0; i < outputs_shape_.size(); ++i) {
      out_shapes[i] = std::make_shared<abstract::TensorShape>(outputs_shape_[i]);
    }
    result = std::make_shared<abstract::TupleShape>(out_shapes);
  } else {
    result = std::make_shared<abstract::TensorShape>(outputs_shape_.front());
  }
  return result;
}

void DvmKernelMod::UpdateInputShapeRef(size_t input_idx, dvm::ShapeRef *ref) { inputs_shape_ref_[input_idx] = ref; }

void DvmKernelMod::DumpRefPair() {
  dump_buf_ << "ref pair: (";
  for (auto it = ref_map_.begin(); it != ref_map_.end(); ++it) {
    if (it != ref_map_.begin()) {
      dump_buf_ << ", ";
    }
    dump_buf_ << "(" << it->first << ", " << it->second << ")";
  }
  dump_buf_ << ")\n";
}

void DvmKernelMod::DumpToFile() {
  std::lock_guard<std::mutex> lock(lock_);
  const std::string dump_dir = "./graph_kernel_dump";
  auto dir_path = FileUtils::CreateNotExistDirs(dump_dir);
  if (!dir_path.has_value()) {
    MS_LOG(INFO) << "Failed to create directory: " << dump_dir;
    return;
  }
  std::string file_name = "dvm_kernel_" + std::to_string(getpid()) + ".txt";
  std::string file_path = dir_path.value() + "/" + file_name;
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream of(file_path, std::ios::app);
  if (!of.is_open()) {
    MS_LOG(INFO) << "Open dump file '" << file_path << "' failed!";
    ChangeFileMode(file_path, S_IRUSR);
    return;
  }
  of << dump_buf_.str() << "\n";
  of.close();
  ChangeFileMode(file_path, S_IRUSR);
  dump_buf_.str("");
}

void SingleDvmKernelMod::CacheLoad(dvm::NDObject *obj, size_t idx) {
  inputs_.push_back(obj);
  inputs_idx_.push_back(idx);
}

void SingleDvmKernelMod::Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) {
  DvmKernelMod::Initialize(inputs_type, outputs_type);
  shapes_ref_source_.reserve(inputs_type.size());
  inputs_.reserve(inputs_type.size());
  outputs_.reserve(outputs_type.size());
  inputs_idx_.reserve(inputs_type.size());
  outputs_idx_.reserve(outputs_type.size());
}

void SingleDvmKernelMod::CacheStore(dvm::NDObject *obj, size_t idx) {
  outputs_.push_back(obj);
  outputs_idx_.push_back(idx);
}

void SingleDvmKernelMod::UpdateIO() {
  inputs_addr_.resize(inputs_.size());
  outputs_addr_.resize(outputs_.size());
  reloc_table_.inputs = inputs_.data();
  reloc_table_.outputs = outputs_.data();
  reloc_table_.inputs_size = inputs_.size();
  reloc_table_.outputs_size = outputs_.size();
}

bool SingleDvmKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (skip_launch_) {
    MS_LOG(DEBUG) << "Skip launch node: " << op_fullname_ << ", because size of some input is zero.";
    return true;
  }
  if (!ref_map_.empty()) {
    CheckRefPair(inputs, outputs);
  }
  for (size_t i = 0; i < inputs_addr_.size(); ++i) {
    inputs_addr_[i] = inputs[inputs_idx_[i]]->device_ptr();
  }
  for (size_t i = 0; i < outputs_addr_.size(); ++i) {
    outputs_addr_[i] = outputs[outputs_idx_[i]]->device_ptr();
  }
  if (profiler::Profiler::GetInstance(kAscendDevice)->GetEnableFlag()) {
    auto ret = kernel_.MsProfLaunch(op_name_.c_str(), op_fullname_.c_str(), reloc_table_, inputs_addr_.data(),
                                    outputs_addr_.data(), workspace.empty() ? nullptr : workspace.back()->device_ptr(),
                                    stream_ptr);
    return ret == 0;
  }
  auto ret = kernel_.Launch(reloc_table_, inputs_addr_.data(), outputs_addr_.data(),
                            workspace.empty() ? nullptr : workspace.back()->device_ptr(), stream_ptr);
  return ret == 0;
}

void SingleDvmKernelMod::UpdateOutputShapes() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto idx = outputs_idx_[i];
    auto shape_ref = kernel_.GetShape(outputs_[i]);
    outputs_shape_[idx] = ShapeVector(shape_ref->data, shape_ref->data + shape_ref->size);
    output_size_list_[idx] = outputs_type_byte_[idx];
    for (auto sh : outputs_shape_[idx]) {
      output_size_list_[idx] *= LongToSize(sh);
    }
  }
}

void ParallelDvmKernelMod::Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) {
  DvmKernelMod::Initialize(inputs_type, outputs_type);
  for (size_t graph_idx = 0; graph_idx < sub_graph_count_; graph_idx++) {
    shapes_ref_source_[graph_idx].reserve(inputs_type.size());
    inputs_[graph_idx].reserve(inputs_type.size());
    outputs_[graph_idx].reserve(outputs_type.size());
    inputs_idx_[graph_idx].reserve(inputs_type.size());
    outputs_idx_[graph_idx].reserve(outputs_type.size());
  }
}

void ParallelDvmKernelMod::CacheLoad(dvm::NDObject *obj, size_t graph_idx, size_t idx) {
  inputs_[graph_idx].push_back(obj);
  inputs_idx_[graph_idx].push_back(idx);
}

void ParallelDvmKernelMod::CacheStore(dvm::NDObject *obj, size_t graph_idx, size_t idx) {
  outputs_[graph_idx].push_back(obj);
  outputs_idx_[graph_idx].push_back(idx);
}

void ParallelDvmKernelMod::UpdateIO() {
  for (size_t i = 0; i < sub_graph_count_; i++) {
    for (size_t j = 0; j < inputs_[i].size(); j++) {
      all_inputs_.push_back(inputs_[i][j]);
      inputs_map_.push_back(inputs_idx_[i][j]);
    }
    for (size_t j = 0; j < outputs_[i].size(); j++) {
      all_outputs_.push_back(outputs_[i][j]);
      outputs_map_.push_back(outputs_idx_[i][j]);
    }
  }
  inputs_addr_.resize(all_inputs_.size());
  outputs_addr_.resize(all_outputs_.size());
  reloc_table_.inputs = all_inputs_.data();
  reloc_table_.outputs = all_outputs_.data();
  reloc_table_.inputs_size = all_inputs_.size();
  reloc_table_.outputs_size = all_outputs_.size();
}

bool ParallelDvmKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (skip_launch_) {
    MS_LOG(EXCEPTION) << "Unexpected behavior in parallel fusion: for node: " << op_fullname_
                      << ", size of some input is zero.";
  }
  if (!ref_map_.empty()) {
    CheckRefPair(inputs, outputs);
  }
  for (size_t i = 0; i < inputs_map_.size(); i++) {
    inputs_addr_[i] = inputs[inputs_map_[i]]->device_ptr();
  }
  for (size_t i = 0; i < outputs_map_.size(); i++) {
    outputs_addr_[i] = outputs[outputs_map_[i]]->device_ptr();
  }
  if (profiler::Profiler::GetInstance(kAscendDevice)->GetEnableFlag()) {
    auto ret = kernel_.MsProfLaunch(op_name_.c_str(), op_fullname_.c_str(), reloc_table_, inputs_addr_.data(),
                                    outputs_addr_.data(), nullptr, stream_ptr);
    return ret == 0;
  }
  auto ret = kernel_.Launch(reloc_table_, inputs_addr_.data(), outputs_addr_.data(), nullptr, stream_ptr);
  return ret == 0;
}

void ParallelDvmKernelMod::UpdateOutputShapes() {
  for (size_t graph_idx = 0; graph_idx < sub_graph_count_; ++graph_idx) {
    for (size_t i = 0; i < outputs_[graph_idx].size(); ++i) {
      auto idx = outputs_idx_[graph_idx][i];
      auto shape_ref = kernel_.GetShape(outputs_[graph_idx][i]);
      outputs_shape_[idx] = ShapeVector(shape_ref->data, shape_ref->data + shape_ref->size);
      output_size_list_[idx] = outputs_type_byte_[idx];
      for (auto sh : outputs_shape_[idx]) {
        output_size_list_[idx] *= LongToSize(sh);
      }
    }
  }
}

}  // namespace kernel
}  // namespace mindspore
