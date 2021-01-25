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

#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#include <set>
#include <map>
#include <string>
#include "src/runtime/opencl/opencl_executor.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;

OpenCLSubGraph::~OpenCLSubGraph() { UnInit(); }

void OpenCLSubGraph::ReplaceOutTensorAndKernelToNull(const std::vector<lite::Tensor *> &in_tensors,
                                                     const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                                                     MemType mem_type) {
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    for (auto &jv : in_kernels.at(i)) {
      MS_ASSERT(jv);
      auto tensors = (mem_type == MemType::IMG) ? jv->in_tensors() : jv->out_tensors();
      auto ft = std::find_if(tensors.begin(), tensors.end(),
                             [&in_tensors, &i](lite::Tensor *kv) { return kv == in_tensors.at(i); });
      if (ft != tensors.end()) {
        *ft = nullptr;
      }
      auto kernels = (mem_type == MemType::IMG) ? jv->in_kernels() : jv->out_kernels();
      std::replace_if(
        kernels.begin(), kernels.end(),
        [this, &in_tensors, &i](kernel::LiteKernel *kv) {
          MS_ASSERT(kv);
          return std::find_if(kv->in_tensors().begin(), kv->in_tensors().end(),
                              [&in_tensors, &i](lite::Tensor *xv) { return xv == in_tensors.at(i); }) !=
                   kv->in_tensors().end() &&
                 this->nodes_set_.count(kv) == 0;
        },
        nullptr);
      if (mem_type == MemType::IMG) {
        jv->set_in_tensors(tensors);
        jv->set_in_kernels(kernels);
      } else {
        jv->set_out_tensors(tensors);
        jv->set_out_kernels(kernels);
      }
    }
  }
}

void OpenCLSubGraph::ReplaceOutTensorAndKernelToConvert(const lite::Tensor *in_tensor,
                                                        const std::vector<kernel::LiteKernel *> &in_kernels,
                                                        lite::Tensor *new_tensor, kernel::LiteKernel *in_convert_op,
                                                        MemType mem_type) {
  MS_ASSERT(in_convert_op);
  auto in_opencl_op = reinterpret_cast<OpenCLKernel *>(in_convert_op);
  for (auto &iv : in_kernels) {
    MS_ASSERT(iv);
    auto kernels = (mem_type == MemType::IMG) ? iv->in_kernels() : iv->out_kernels();
    auto fk = std::find_if(kernels.begin(), kernels.end(), [&](kernel::LiteKernel *kv) { return kv == nullptr; });
    if (fk != kernels.end()) {
      *fk = in_convert_op;
    } else {
      kernels.emplace_back(in_convert_op);
    }
    auto tensors = (mem_type == MemType::IMG) ? iv->in_tensors() : iv->out_tensors();
    auto ft = std::find_if(tensors.begin(), tensors.end(), [&](lite::Tensor *kv) { return kv == nullptr; });
    if (ft != tensors.end()) {
      *ft = new_tensor;
    } else {
      tensors.emplace_back(new_tensor);
    }
    if (mem_type == MemType::IMG) {
      iv->set_in_kernels(kernels);
      iv->set_in_tensors(tensors);
      in_opencl_op->AddOutKernel(iv);
    } else {
      iv->set_out_kernels(kernels);
      iv->set_out_tensors(tensors);
      in_convert_op->AddInKernel(iv);
    }
  }
}

int OpenCLSubGraph::GenToFormatOp(const std::vector<lite::Tensor *> &in_tensors,
                                  const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                                  std::vector<lite::Tensor *> *out_tensors,
                                  std::vector<OpenCLToFormatParameter *> *out_parameters,
                                  std::vector<LiteKernel *> *out_convert_ops, MemType mem_type) {
  MS_ASSERT(out_tensors);
  MS_ASSERT(out_parameters);
  MS_ASSERT(out_convert_ops);
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  std::vector<std::vector<kernel::LiteKernel *>> loop_kernels;
  if (mem_type == MemType::BUF) {
    GetKernelFromToTensor(in_tensors, nodes_, &loop_kernels, true);
  }

  ReplaceOutTensorAndKernelToNull(in_tensors, in_kernels, mem_type);

  for (size_t i = 0; i < in_tensors.size(); ++i) {
    auto *in_tensor = in_tensors.at(i);
    auto dst_format = (mem_type == MemType::IMG) ? schema::Format::Format_NHWC4 : schema::Format::Format_NHWC;
    auto src_format = (mem_type == MemType::IMG) ? schema::Format::Format_NHWC : schema::Format::Format_NHWC4;
    auto *new_tensor = new (std::nothrow)
      lite::Tensor(in_tensor->data_type(), in_tensor->shape(), in_tensor->format(), lite::Tensor::VAR);
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new tensor failed!";
      return RET_ERROR;
    }
    if (mem_type == MemType::IMG) {
      new_tensor->set_format(dst_format);
      in_tensor->set_format(src_format);
    } else {
      new_tensor->set_format(src_format);
      in_tensor->set_format(dst_format);
    }

    out_tensors->emplace_back(new_tensor);
    KernelKey desc{kGPU, kNumberTypeFloat32, schema::PrimitiveType_ToFormat};
    if (mem_type == MemType::IMG && ocl_runtime_->GetFp16Enable()) {
      desc.data_type = kNumberTypeFloat16;
      new_tensor->set_data_type(kNumberTypeFloat16);
    }
    auto *parameter = static_cast<OpenCLToFormatParameter *>(malloc(sizeof(OpenCLToFormatParameter)));
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new parameter failed!";
      delete new_tensor;
      new_tensor = nullptr;
      return RET_ERROR;
    }
    parameter->op_parameter.type_ = mindspore::schema::PrimitiveType_ToFormat;
    parameter->src_format = src_format;
    parameter->dst_format = dst_format;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op = nullptr;
    if (mem_type == MemType::IMG) {
      in_convert_op =
        lite::GetOpenCLKernel({in_tensor}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter), context_, desc);
    } else {
      in_convert_op =
        lite::GetOpenCLKernel({new_tensor}, {in_tensor}, reinterpret_cast<OpParameter *>(parameter), context_, desc);
    }
    MS_ASSERT(in_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    static int index = 0;
    in_convert_op->set_name("ToFormat_" + std::to_string(index));

    ReplaceOutTensorAndKernelToConvert(in_tensor, in_kernels.at(i), new_tensor, in_convert_op, mem_type);

    // replace in_tensor of inner kernel which use out tensor
    if (mem_type == MemType::BUF) {
      for (auto &iv : loop_kernels[i]) {
        MS_ASSERT(iv);
        auto tensors = iv->in_tensors();
        auto jv = std::find(tensors.begin(), tensors.end(), in_tensors.at(i));
        if (jv != tensors.end()) {
          *jv = new_tensor;
          iv->set_in_tensors(tensors);
        }
      }
    }

    out_convert_ops->emplace_back(in_convert_op);
  }
  return RET_OK;
}
int OpenCLSubGraph::InsertOpsPass() {
  GetInOutNodes();

  std::vector<std::vector<kernel::LiteKernel *>> from_kernels_;
  GetKernelFromToTensor(in_tensors_, in_nodes_, &from_kernels_, true);
  int ret =
    GenToFormatOp(in_tensors_, from_kernels_, &in_convert_tensors_, &in_parameters_, &in_convert_ops_, MemType::IMG);
  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.begin(), in_convert_ops_.begin(), in_convert_ops_.end());

  std::vector<std::vector<kernel::LiteKernel *>> to_kernels_;
  GetKernelFromToTensor(out_tensors_, out_nodes_, &to_kernels_, false);
  ret =
    GenToFormatOp(out_tensors_, to_kernels_, &out_convert_tensors_, &out_parameters_, &out_convert_ops_, MemType::BUF);
  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.end(), out_convert_ops_.begin(), out_convert_ops_.end());
  GetInOutNodes();
  return RET_OK;
}
int OpenCLSubGraph::Init() {
  allocator_ = ocl_runtime_->GetAllocator();
  MS_LOG(DEBUG) << "input num=" << in_tensors_.size() << ", output num=" << out_tensors_.size();
  for (const auto tensor : in_tensors_) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors_) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  std::map<std::string, std::function<int(void)>> pass_manager{
    {"InsertOpsPass", std::bind(&OpenCLSubGraph::InsertOpsPass, this)},
    {"UpdateTensorDataTypePass", std::bind(&OpenCLSubGraph::UpdateTensorDataTypePass, this)},
    {"FusionPass", std::bind(&OpenCLSubGraph::FusionPass, this)}};
  for (auto iv : pass_manager) {
    auto ret = iv.second();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run Pass: " << iv.first << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int OpenCLSubGraph::UpdateTensorDataTypePass() {
  bool is_fp16 = ocl_runtime_->GetFp16Enable();
  MS_ASSERT(in_tensors_[0]);
  if (is_fp16 && (in_tensors_[0]->data_type() == kNumberTypeFloat32)) {
    std::set<lite::Tensor *> out_set;
    out_set.insert(in_tensors_.begin(), in_tensors_.end());
    out_set.insert(out_tensors_.begin(), out_tensors_.end());
    for (auto iv : nodes_) {
      MS_ASSERT(iv);
      auto cur_outs = iv->out_tensors();
      for (auto jv : cur_outs) {
        if (out_set.count(jv) == 0) {
          MS_ASSERT(jv);
          // if Fp16Enable, only change fp32 to fp16, other dtype is reserved
          if (jv->data_type() == kNumberTypeFloat32) {
            jv->set_data_type(kNumberTypeFloat16);
          }
        }
      }
    }
  }
  return RET_OK;
}

void OpenCLSubGraph::GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                                           const std::vector<kernel::LiteKernel *> &in_kernels,
                                           std::vector<std::vector<kernel::LiteKernel *>> *out_kernels, bool is_from) {
  std::vector<std::set<lite::Tensor *>> ksets;
  for (auto jv : in_kernels) {
    MS_ASSERT(jv);
    auto tens = is_from ? jv->in_tensors() : jv->out_tensors();
    std::set<lite::Tensor *> kset;
    kset.insert(tens.begin(), tens.end());
    ksets.emplace_back(kset);
  }
  MS_ASSERT(out_kernels);
  for (auto in_tensor : in_tensors) {
    std::vector<kernel::LiteKernel *> kvec;
    for (size_t j = 0; j < in_kernels.size(); ++j) {
      if (ksets[j].count(in_tensor)) {
        kvec.emplace_back(in_kernels[j]);
      }
    }
    out_kernels->emplace_back(kvec);
  }
}

void OpenCLSubGraph::GetInOutNodes() {
  this->in_nodes_.clear();
  this->out_nodes_.clear();
  for (auto *node : nodes_) {
    for (auto *tensor : node->in_tensors()) {
      if (std::find(in_tensors_.begin(), in_tensors_.end(), tensor) != in_tensors_.end()) {
        in_nodes_.emplace_back(node);
        break;
      }
    }
    for (auto *tensor : node->out_tensors()) {
      if (std::find(out_tensors_.begin(), out_tensors_.end(), tensor) != out_tensors_.end()) {
        out_nodes_.emplace_back(node);
        break;
      }
    }
  }
}

bool OpenCLSubGraph::IsSubGraphInferShapeDone() {
  for (auto node : this->nodes_) {
    auto opencl_kernel = reinterpret_cast<kernel::OpenCLKernel *>(node);
    if (!opencl_kernel->GetInferShapeFlag()) {
      return false;
    }
  }
  return true;
}

int OpenCLSubGraph::Prepare() {
  executor_ = new (std::nothrow) lite::opencl::OpenCLExecutor();
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "Create OpenCLExecutor fail";
    return RET_ERROR;
  }
  auto ret = RET_OK;
  for (auto node : this->nodes_) {
    if (node == nullptr) {
      MS_LOG(ERROR) << "node in Subgraph is nullptr";
      return mindspore::lite::RET_NULL_PTR;
    }
    auto opencl_kernel = reinterpret_cast<kernel::OpenCLKernel *>(node);
    std::set<int> pre_init_weight_list = {schema::PrimitiveType_MatMul, schema::PrimitiveType_BiasAdd};
    if (pre_init_weight_list.find(opencl_kernel->Type()) != pre_init_weight_list.end()) {
      ret = opencl_kernel->InitWeights();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "init weights " << node->name() << " failed";
        return ret;
      }
    }
    if (opencl_kernel->GetInferShapeFlag()) {
      ret = node->Prepare();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "prepare node " << node->name() << " failed";
        return ret;
      }
    }
  }
  auto opencl_exec = reinterpret_cast<lite::opencl::OpenCLExecutor *>(executor_);
  // If tuning_mode is DEFAULT, just malloc memory for reuse.
  ret = opencl_exec->RunOrTune(in_tensors_, out_tensors_, nodes_, allocator_, nullptr, nullptr, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  return RET_OK;
}

void OpenCLSubGraph::UnInit() {
  for (const auto &tensor : in_convert_tensors_) {
    delete tensor;
  }
  in_convert_tensors_.clear();
  for (const auto &tensor : out_convert_tensors_) {
    delete tensor;
  }
  out_convert_tensors_.clear();
  for (const auto &op : nodes_) {
    delete op;
  }
  nodes_.clear();
  in_convert_ops_.clear();
  out_convert_ops_.clear();
  delete this->executor_;
}

int OpenCLSubGraph::ReSize() { return ReSize(false); }

int OpenCLSubGraph::ReSize(bool interrupt) {
  for (auto kernel : nodes_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    auto opencl_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "all nodes in should be kernel";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> inputs = kernel->in_tensors();
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
    }
    opencl_kernel->SetInferShapeFlag(false);
  }
  for (auto kernel : nodes_) {
    auto opencl_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto ret = opencl_kernel->ReSize();
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "ReSize " << opencl_kernel->name() << "failed!";
      if (interrupt) {
        return ret;
      } else {
        break;
      }
    }
  }
  return RET_OK;
}

int OpenCLSubGraph::Run() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  int ret;
  for (auto &tensor : in_tensors_) {
    MS_ASSERT(tensor);
    if (tensor->data_c() == nullptr) {
      MS_LOG(ERROR) << "OpenCL subgraph input tensor data is null";
      return RET_ERROR;
    }
    ret = allocator_->UnmapBuffer(tensor->data_c());
    if (ret != RET_OK) {
      return ret;
    }
  }

  ret = executor_->Run(in_tensors_, out_tensors_, nodes_, allocator_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  if (!ocl_runtime_->SyncCommandQueue()) {
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
