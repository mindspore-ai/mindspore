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

#include "src/litert/kernel/opencl/opencl_subgraph.h"
#include <set>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "src/litert/kernel/gpu/opencl/opencl_executor.h"
#include "src/litert/kernel/opencl/utils.h"
#include "src/litert/kernel/opencl/kernel/to_format.h"
#include "src/litert/kernel/opencl/kernel/gl_to_cl.h"
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using PrimType::PrimType_Inner_ToFormat;

OpenCLSubGraph::~OpenCLSubGraph() { UnInit(); }

void OpenCLSubGraph::ReplaceOutTensorAndKernelToConvert(const lite::Tensor *in_tensor,
                                                        const std::vector<kernel::KernelExec *> &in_kernels,
                                                        lite::Tensor *new_tensor, kernel::KernelExec *in_convert_op,
                                                        MemType mem_type) {
  MS_ASSERT(in_convert_op);
  auto in_opencl_op = in_convert_op;
  for (auto &iv : in_kernels) {
    MS_ASSERT(iv);
    auto kernels = (mem_type == MemType::IMG) ? iv->in_kernels() : iv->out_kernels();
    auto fk = std::find_if(kernels.begin(), kernels.end(), [&](kernel::KernelExec *kv) { return kv == iv; });
    if (fk != kernels.end()) {
      *fk = in_convert_op;
    } else {
      kernels.emplace_back(in_convert_op);
    }
    auto tensors = (mem_type == MemType::IMG) ? iv->in_tensors() : iv->out_tensors();
    auto ft = std::find_if(tensors.begin(), tensors.end(), [&](lite::Tensor *kv) { return kv == in_tensor; });
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
                                  const std::vector<std::vector<kernel::KernelExec *>> &in_kernels,
                                  std::vector<lite::Tensor *> *out_tensors,
                                  std::vector<OpenCLToFormatParameter *> *out_parameters,
                                  std::vector<KernelExec *> *out_convert_ops, MemType mem_type) {
  MS_ASSERT(out_tensors);
  MS_ASSERT(out_parameters);
  MS_ASSERT(out_convert_ops);
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  std::vector<std::vector<kernel::KernelExec *>> loop_kernels;
  if (mem_type == MemType::BUF) {
    GetKernelFromToTensor(in_tensors, nodes_, &loop_kernels, true);
  }

  for (size_t i = 0; i < in_tensors.size(); ++i) {
    auto *in_tensor = in_tensors.at(i);
    auto *new_tensor = new (std::nothrow)
      lite::Tensor(in_tensor->data_type(), in_tensor->shape(), in_tensor->format(), lite::Category::VAR);
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new tensor failed!";
      return RET_ERROR;
    }
    for (const auto &param : in_tensor->quant_params()) {
      new_tensor->AddQuantParam(param);
    }

    out_tensors->emplace_back(new_tensor);
    KernelKey desc{kGPU, kNumberTypeFloat32, NHWC, PrimType_Inner_ToFormat};
    auto *parameter = static_cast<OpenCLToFormatParameter *>(malloc(sizeof(OpenCLToFormatParameter)));
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new parameter failed!";
      delete new_tensor;
      new_tensor = nullptr;
      return RET_ERROR;
    }

    parameter->op_parameter.is_zero_shape_ = false;
    parameter->op_parameter.type_ = PrimType_Inner_ToFormat;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op_inner = nullptr;
    if (mem_type == MemType::IMG) {
      in_convert_op_inner = OpenCLKernelCreator<ToFormatOpenCLKernel>(
        {in_tensor}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter), this->Context(), desc);
    } else {
      in_convert_op_inner = OpenCLKernelCreator<ToFormatOpenCLKernel>(
        {new_tensor}, {in_tensor}, reinterpret_cast<OpParameter *>(parameter), this->Context(), desc);
    }
    MS_ASSERT(in_convert_op_inner);
    if (in_convert_op_inner == nullptr ||
        reinterpret_cast<ToFormatOpenCLKernel *>(in_convert_op_inner)->CheckSpecs() != RET_OK) {
      MS_LOG(ERROR) << "OpenCLSubGraph create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    std::shared_ptr<kernel::Kernel> inner_convert_op(in_convert_op_inner);
    auto *in_convert_op = new (std::nothrow) kernel::KernelExec(inner_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    static int index = 0;
    in_convert_op->set_name("ToFormat_" + std::to_string(index++));

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

int OpenCLSubGraph::GenGLToCLOp(const std::vector<lite::Tensor *> &in_tensors,
                                const std::vector<std::vector<kernel::KernelExec *>> &in_kernels,
                                std::vector<lite::Tensor *> *out_tensors,
                                std::vector<OpenGLTexture2DToOpenCLParameter *> *out_parameters,
                                std::vector<KernelExec *> *out_convert_ops, MemType mem_type) {
  MS_ASSERT(out_tensors);
  MS_ASSERT(out_parameters);
  MS_ASSERT(out_convert_ops);
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  std::vector<std::vector<kernel::KernelExec *>> loop_kernels;
  if (mem_type == MemType::GLTexture) {
    GetKernelFromToTensor(in_tensors, nodes_, &loop_kernels, true);
  }

  for (size_t i = 0; i < in_tensors.size(); ++i) {
    auto *in_tensor = in_tensors.at(i);
    auto *new_tensor = new (std::nothrow)
      lite::Tensor(in_tensor->data_type(), in_tensor->shape(), in_tensor->format(), lite::Category::VAR);
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new tensor failed!";
      return RET_ERROR;
    }
    for (const auto &param : in_tensor->quant_params()) {
      new_tensor->AddQuantParam(param);
    }

    out_tensors->emplace_back(new_tensor);
    KernelKey desc{kGPU, kNumberTypeGLUInt, NHWC, PrimType::PrimType_Inner_GltextureToOpencl};
    auto *parameter = static_cast<OpenGLTexture2DToOpenCLParameter *>(malloc(sizeof(OpenGLTexture2DToOpenCLParameter)));
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph new parameter failed!";
      delete new_tensor;
      new_tensor = nullptr;
      return RET_ERROR;
    }

    parameter->op_parameter.is_zero_shape_ = false;
    parameter->op_parameter.type_ = PrimType::PrimType_Inner_GltextureToOpencl;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op_inner = nullptr;
    if (mem_type == MemType::IMG) {
      in_convert_op_inner = OpenCLKernelCreator<GLToCLOpenCLKernel>(
        {in_tensor}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter), this->Context(), desc);
    } else {
      in_convert_op_inner = OpenCLKernelCreator<GLToCLOpenCLKernel>(
        {new_tensor}, {in_tensor}, reinterpret_cast<OpParameter *>(parameter), this->Context(), desc);
    }
    MS_ASSERT(in_convert_op_inner);
    if (in_convert_op_inner == nullptr ||
        reinterpret_cast<GLToCLOpenCLKernel *>(in_convert_op_inner)->CheckSpecs() != RET_OK) {
      MS_LOG(ERROR) << "OpenCLSubGraph create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    std::shared_ptr<kernel::Kernel> inner_convert_op(in_convert_op_inner);
    auto *in_convert_op = new (std::nothrow) kernel::KernelExec(inner_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "OpenCLSubGraph create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    static int index = 0;
    in_convert_op->set_name("GLToCL_" + std::to_string(index++));
    ReplaceOutTensorAndKernelToConvert(in_tensor, in_kernels.at(i), new_tensor, in_convert_op, mem_type);
    // replace in_tensor of inner kernel which use out tensor
    if (mem_type == MemType::GLTexture) {
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

  std::vector<std::vector<kernel::KernelExec *>> from_kernels_;
  GetKernelFromToTensor(in_tensors(), in_nodes_, &from_kernels_, true);
  int ret = 0;

  if (this->GetOpenGLTextureEnable() == true) {
    ret = GenGLToCLOp(in_tensors(), from_kernels_, &in_convert_tensors_, &gl_in_parameters_, &in_convert_ops_,
                      MemType::IMG);
  } else {
    ret =
      GenToFormatOp(in_tensors(), from_kernels_, &in_convert_tensors_, &in_parameters_, &in_convert_ops_, MemType::IMG);
  }

  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.begin(), in_convert_ops_.begin(), in_convert_ops_.end());

  std::vector<std::vector<kernel::KernelExec *>> to_kernels_;
  GetKernelFromToTensor(out_tensors(), out_nodes_, &to_kernels_, false);

  if (this->GetOpenGLTextureEnable()) {
    ret = GenGLToCLOp(out_tensors(), to_kernels_, &out_convert_tensors_, &gl_out_parameters_, &out_convert_ops_,
                      MemType::GLTexture);
  } else {
    ret = GenToFormatOp(out_tensors(), to_kernels_, &out_convert_tensors_, &out_parameters_, &out_convert_ops_,
                        MemType::BUF);
  }

  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.end(), out_convert_ops_.begin(), out_convert_ops_.end());
  GetInOutNodes();
  return RET_OK;
}

int OpenCLSubGraph::RunPass() {
  // The fp16 operator in heterogeneous scenes needs to be set to fp32
  // to prevent the frame from being converted to fp16 in advance.
  auto in_first_tensor = in_tensors().front();
  if (in_first_tensor->IsGraphInput() &&
      (in_first_tensor->data_type() == kNumberTypeFloat32 || in_first_tensor->data_type() == kNumberTypeFloat16)) {
    desc_.data_type = in_tensors()[0]->data_type();
  }
  allocator_ = ocl_runtime_->GetAllocator();
  MS_LOG(DEBUG) << "input num=" << in_tensors().size() << ", output num=" << out_tensors().size();
  for (const auto tensor : in_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  std::vector<std::pair<std::string, std::function<int(void)>>> pass_manager{
    {"FusionPass", std::bind(&OpenCLSubGraph::FusionPass, this)},
    {"InsertOpsPass", std::bind(&OpenCLSubGraph::InsertOpsPass, this)},
    {"UpdateTensorDataTypePass", std::bind(&OpenCLSubGraph::UpdateTensorDataTypePass, this)},
  };
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
  if (is_fp16 && subgraph_type() == kGpuFp16SubGraph) {
    auto in_tensors = this->in_tensors();
    auto out_tensors = this->out_tensors();
    for (auto iv : nodes_) {
      MS_ASSERT(iv);
      auto cur_outs = iv->out_tensors();
      // if softmax is last kernel, output fp32 tensor
      if (iv->type() == schema::PrimitiveType_Softmax) {
        bool last_kernel = true;
        for (auto k : iv->out_kernels()) {
          int type = k->op_parameter() == nullptr ? k->type() : k->op_parameter()->type_;
          if (type == PrimType::PrimType_Inner_ToFormat) {
            last_kernel = false;
            break;
          }
        }
        if (last_kernel) continue;
      }
      for (auto jv : cur_outs) {
        MS_ASSERT(jv);
        // if Fp16Enable, only change fp32 to fp16, other dtype is reserved
        if (jv->data_type() == kNumberTypeFloat32 && !jv->IsGraphOutput()) {
          jv->set_data_type(kNumberTypeFloat16);
        }
      }
    }
  }
  return RET_OK;
}

void OpenCLSubGraph::GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                                           const std::vector<kernel::KernelExec *> &in_kernels,
                                           std::vector<std::vector<kernel::KernelExec *>> *out_kernels, bool is_from) {
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
    std::vector<kernel::KernelExec *> kvec;
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
  auto in_tensors = this->in_tensors();
  auto out_tensors = this->out_tensors();
  for (auto *node : nodes_) {
    for (auto *tensor : node->in_tensors()) {
      if (std::find(in_tensors.begin(), in_tensors.end(), tensor) != in_tensors.end()) {
        in_nodes_.emplace_back(node);
        break;
      }
    }
    for (auto *tensor : node->out_tensors()) {
      if (std::find(out_tensors.begin(), out_tensors.end(), tensor) != out_tensors.end()) {
        out_nodes_.emplace_back(node);
        break;
      }
    }
  }
}

int OpenCLSubGraph::Prepare() {
  ocl_runtime_->SetFp16Enable(subgraph_type() == kGpuFp16SubGraph);

  for (const auto tensor : in_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors()) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  executor_ = new (std::nothrow) lite::opencl::OpenCLExecutor();
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "Create OpenCLExecutor fail";
    return RET_ERROR;
  }
  for (auto node : this->nodes_) {
    if (node == nullptr) {
      MS_LOG(ERROR) << "node in Subgraph is nullptr";
      return mindspore::lite::RET_NULL_PTR;
    }
    for (const auto tensor : node->out_tensors()) {
      CHECK_NULL_RETURN(tensor);
      MS_CHECK_TRUE_RET(tensor->data() == nullptr, RET_ERROR);
      tensor->set_allocator(allocator_);
    }
    if (desc_.provider == kBuiltin) {
      auto opencl_kernel = reinterpret_cast<kernel::OpenCLKernel *>(node->kernel());
      std::set<int> pre_init_weight_list = {schema::PrimitiveType_MatMulFusion, schema::PrimitiveType_BiasAdd};
      if (pre_init_weight_list.find(opencl_kernel->type()) != pre_init_weight_list.end()) {
        auto ret = opencl_kernel->InitWeights();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "init weights " << node->name() << " failed";
          return ret;
        }
      }
    }
    if (node->InferShapeDone()) {
      auto ret = node->Prepare();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "prepare node " << node->name() << " failed";
        return ret;
      }
    }
  }
  if (all_kernels_infer_done_) {
    auto opencl_exec = reinterpret_cast<lite::opencl::OpenCLExecutor *>(executor_);
    // If tuning_mode is DEFAULT, just malloc memory for reuse.
    auto ret = opencl_exec->RunOrTune(in_tensors(), out_tensors(), nodes_, nullptr, nullptr, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run opencl Tuning failed: " << ret;
      return ret;
    }
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

int OpenCLSubGraph::ReSize() {
  for (auto kernel : nodes_) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "all nodes in should be kernel";
      return RET_ERROR;
    }
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
      output->set_shape({-1});
    }
  }
  for (auto kernel : nodes_) {
    auto ret = kernel->ReSize();
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "ReSize " << kernel->name() << "failed!, ret:" << ret;
      return ret;
    }
  }
  return RET_OK;
}

int OpenCLSubGraph::Execute() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  int ret;
  for (auto &tensor : in_tensors()) {
    MS_ASSERT(tensor);
    if (tensor->data() == nullptr) {
      MS_LOG(ERROR) << "OpenCL subgraph input tensor data is null";
      return RET_ERROR;
    }
    ret = allocator_->UnmapBuffer(tensor->data());
    if (ret != RET_OK) {
      return ret;
    }
  }

  ret = executor_->Run(in_tensors(), out_tensors(), nodes_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  if (!ocl_runtime_->SyncCommandQueue()) {
    MS_LOG(ERROR) << "SyncCommandQueue failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int OpenCLSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  int ret;
  for (auto &tensor : in_tensors()) {
    MS_ASSERT(tensor);
    if (tensor->data() == nullptr) {
      MS_LOG(ERROR) << "OpenCL subgraph input tensor data is null";
      return RET_ERROR;
    }
    ret = allocator_->UnmapBuffer(tensor->data());
    if (ret != RET_OK) {
      return ret;
    }
  }

  ret = executor_->Run(in_tensors(), out_tensors(), nodes_, before, after);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  if (!ocl_runtime_->SyncCommandQueue()) {
    MS_LOG(ERROR) << "SyncCommandQueue failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
