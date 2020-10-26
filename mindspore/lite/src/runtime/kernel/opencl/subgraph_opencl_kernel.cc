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

#include "src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include <set>
#include "src/runtime/opencl/opencl_executor.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
SubGraphOpenCLKernel::~SubGraphOpenCLKernel() { UnInit(); }

int SubGraphOpenCLKernel::GenToFormatOp(const std::vector<lite::Tensor *> &in_tensors,
                                        const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                                        std::vector<lite::Tensor *> *out_tensors,
                                        std::vector<OpenCLToFormatParameter *> *out_parameters,
                                        std::vector<LiteKernel *> *out_convert_ops, OpenCLMemType mem_type) {
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  MS_ASSERT(in_tensors.size() == to_kernels.size());
  MS_ASSERT(in_tensors.size() == from_kernels.size());
  for (auto &iv : in_kernels) {
    for (auto &jv : iv) {
      if (mem_type == OpenCLMemType::IMG) {
        jv->set_in_tensors({});
        jv->SetInKernel({});
      } else {
        jv->set_out_tensors({});
        jv->SetOutKernel({});
      }
    }
  }
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    if (in_tensors.at(i)->shape().size() <= 1) {
      if (mem_type == OpenCLMemType::IMG) {
        for (auto &iv : in_kernels[i]) {
          auto tensors = iv->in_tensors();
          tensors.emplace_back(in_tensors.at(i));
          iv->set_in_tensors(tensors);
        }
      } else {
        for (auto &iv : in_kernels[i]) {
          auto tensors = iv->out_tensors();
          tensors.emplace_back(in_tensors.at(i));
          iv->set_out_tensors(tensors);
        }
      }
      continue;
    }
    auto dst_format = (mem_type == OpenCLMemType::IMG) ? schema::Format::Format_NHWC4 : schema::Format::Format_NHWC;
    auto src_format = (mem_type == OpenCLMemType::IMG) ? schema::Format::Format_NHWC : schema::Format::Format_NHWC4;
    auto *new_tensor = new (std::nothrow) lite::Tensor();
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new tensor failed!";
      return RET_ERROR;
    }
    new_tensor->CopyTensor(*in_tensors[i]);
    if (mem_type == OpenCLMemType::IMG) {
      new_tensor->SetFormat(dst_format);
      in_tensors[i]->SetFormat(src_format);
    } else {
      new_tensor->SetFormat(src_format);
      in_tensors[i]->SetFormat(dst_format);
    }

    out_tensors->emplace_back(new_tensor);
    KernelKey desc{kGPU, kNumberTypeFloat32, schema::PrimitiveType_ToFormat};
    if (mem_type == OpenCLMemType::IMG && ocl_runtime_->GetFp16Enable()) {
      desc.data_type = kNumberTypeFloat16;
      new_tensor->set_data_type(kNumberTypeFloat16);
    }
    auto *parameter = static_cast<OpenCLToFormatParameter *>(malloc(sizeof(OpenCLToFormatParameter)));
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new parameter failed!";
      delete new_tensor;
      new_tensor = nullptr;
      return RET_ERROR;
    }
    parameter->src_format = src_format;
    parameter->dst_format = dst_format;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op = nullptr;
    if (mem_type == OpenCLMemType::IMG) {
      in_convert_op = lite::GetOpenCLKernel({in_tensors[i]}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter),
                                            context_, desc);
    } else {
      in_convert_op = lite::GetOpenCLKernel({new_tensor}, {in_tensors[i]}, reinterpret_cast<OpParameter *>(parameter),
                                            context_, desc);
    }
    MS_ASSERT(in_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }
    auto in_opencl_op = reinterpret_cast<OpenCLKernel *>(in_convert_op);
    if (mem_type == OpenCLMemType::IMG) {
      for (auto &iv : in_kernels[i]) {
        in_opencl_op->AddOutKernel(iv);
        auto kernels = iv->in_kernels();
        kernels.emplace_back(in_convert_op);
        iv->SetInKernel(kernels);
        auto tensors = iv->in_tensors();
        tensors.emplace_back(new_tensor);
        iv->set_in_tensors(tensors);
      }
    } else {
      for (auto &iv : in_kernels[i]) {
        auto kernels = iv->out_kernels();
        kernels.emplace_back(in_convert_op);
        iv->SetOutKernel(kernels);
        auto tensors = iv->out_tensors();
        tensors.emplace_back(new_tensor);
        iv->set_out_tensors(tensors);
        in_convert_op->AddInKernel(iv);
      }
    }
    out_convert_ops->emplace_back(in_convert_op);
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::Init() {
  allocator_ = ocl_runtime_->GetAllocator();
  MS_LOG(DEBUG) << "input num=" << in_tensors_.size() << ", output num=" << out_tensors_.size();
  for (const auto tensor : in_tensors_) {
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors_) {
    tensor->set_allocator(allocator_);
  }

  std::vector<std::vector<kernel::LiteKernel *>> from_kernels_;
  GetKernelFromToTensor(in_tensors_, in_kernels_, &from_kernels_, true);
  int ret = GenToFormatOp(in_tensors_, from_kernels_, &in_convert_tensors_, &in_parameters_, &in_convert_ops_,
                          OpenCLMemType::IMG);
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  nodes_.insert(nodes_.begin(), in_convert_ops_.begin(), in_convert_ops_.end());

  std::vector<std::vector<kernel::LiteKernel *>> to_kernels_;
  GetKernelFromToTensor(out_tensors_, out_kernels_, &to_kernels_, false);
  ret = GenToFormatOp(out_tensors_, to_kernels_, &out_convert_tensors_, &out_parameters_, &out_convert_ops_,
                      OpenCLMemType::BUF);
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  nodes_.insert(nodes_.end(), out_convert_ops_.begin(), out_convert_ops_.end());

  UpdateTensorDataType();

  MallocTensorWithReuse();

  return RET_OK;
}

int SubGraphOpenCLKernel::UpdateTensorDataType() {
  bool is_fp16 = ocl_runtime_->GetFp16Enable();
  if (is_fp16 && (in_tensors_[0]->data_type() == kNumberTypeFloat32)) {
    std::set<lite::Tensor *> out_set;
    out_set.insert(in_tensors_.begin(), in_tensors_.end());
    out_set.insert(out_tensors_.begin(), out_tensors_.end());
    for (auto iv : nodes_) {
      auto cur_outs = iv->out_tensors();
      for (auto jv : cur_outs) {
        if (out_set.count(jv) == 0) {
          jv->set_data_type(kNumberTypeFloat16);
        }
      }
    }
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::MallocTensorWithReuse() {
  kernel::LiteKernelUtil::InitTensorRefCount(nodes_);
  for (auto *kernel : nodes_) {
    MS_ASSERT(nullptr != kernel);
    auto *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto outputs = kernel->out_tensors();
    for (auto i = 0; i < outputs.size(); ++i) {
      auto *output = outputs.at(i);
      MS_ASSERT(nullptr != output);
      if (op_kernel->GetMemType() == OpenCLMemType::IMG) {
        std::vector<size_t> img_size;
        op_kernel->GetImageSize(i, &img_size);
        auto data_ptr = allocator_->Malloc(output->Size(), img_size);
        output->set_data(data_ptr);
      } else {
        output->MallocData(allocator_);
      }
      output->set_allocator(allocator_);
    }
    for (auto input_kernel : kernel->in_kernels()) {
      MS_ASSERT(nullptr != input_kernel);
      auto ret = input_kernel->DecOutTensorRefCount();
      if (0 != ret) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
      }
    }
  }
  for (auto kernel : out_kernels_) {
    MS_ASSERT(nullptr != kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  for (auto kernel : in_convert_ops_) {
    MS_ASSERT(nullptr != kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  for (auto kernel : out_convert_ops_) {
    MS_ASSERT(nullptr != kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                                                const std::vector<kernel::LiteKernel *> &in_kernels,
                                                std::vector<std::vector<kernel::LiteKernel *>> *out_kernels,
                                                bool is_from) {
  std::vector<std::set<lite::Tensor *>> ksets;
  for (auto jv : in_kernels) {
    auto tens = is_from ? jv->in_tensors() : jv->out_tensors();
    std::set<lite::Tensor *> kset;
    kset.insert(tens.begin(), tens.end());
    ksets.emplace_back(kset);
  }
  for (auto in_tensor : in_tensors) {
    std::vector<kernel::LiteKernel *> kvec;
    for (size_t j = 0; j < in_kernels.size(); ++j) {
      if (ksets[j].count(in_tensor)) {
        kvec.emplace_back(in_kernels[j]);
      }
    }
    out_kernels->emplace_back(kvec);
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::UnInit() {
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
  return RET_OK;
}

int SubGraphOpenCLKernel::InferShape() { return RET_OK; }

int SubGraphOpenCLKernel::ReSize() { return RET_OK; }

int SubGraphOpenCLKernel::Run() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  for (auto &tensor : in_tensors_) {
    if (tensor->data_c() == nullptr) {
      MS_LOG(ERROR) << "OpenCL subgraph input tensor data is null";
      return RET_ERROR;
    }
    allocator_->UnmapBuffer(tensor->data_c());
  }

  auto ret = executor_->Run(in_tensors_, out_tensors_, nodes_, allocator_);
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  ocl_runtime_->SyncCommandQueue();

  return RET_OK;
}
}  // namespace mindspore::kernel
