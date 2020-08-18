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
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {

SubGraphOpenCLKernel::~SubGraphOpenCLKernel() { UnInit(); }

int SubGraphOpenCLKernel::GenToFormatOp(const std::vector<lite::tensor::Tensor *> &in_tensors,
                                        const std::vector<std::vector<kernel::LiteKernel *>> in_kernels,
                                        std::vector<lite::tensor::Tensor *> *out_tensors,
                                        std::vector<OpenCLToFormatParameter *> *out_parameters,
                                        std::vector<LiteKernel *> *out_convert_ops, OpenCLMemType mem_type) {
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  MS_ASSERT(in_tensors.size() == to_kernels.size());
  MS_ASSERT(in_tensors.size() == from_kernels.size());
  for (auto &iv : in_kernels) {
    for (auto &jv : iv) {
      OpenCLKernel *cur_opencl_op = reinterpret_cast<OpenCLKernel *>(jv);
      schema::Format ori_format = cur_opencl_op->GetOriFormat();
      auto tens = cur_opencl_op->out_tensors();
      if (mem_type == OpenCLMemType::BUF && mem_type == cur_opencl_op->GetMemType() &&
          tens[0]->GetFormat() == ori_format) {
        continue;
      }
      if (mem_type == OpenCLMemType::IMG) {
        jv->set_in_tensors({});
      } else {
        jv->set_out_tensors({});
      }
    }
  }
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    OpenCLKernel *cur_opencl_op = reinterpret_cast<OpenCLKernel *>(in_kernels[i][0]);
    schema::Format ori_format = cur_opencl_op->GetOriFormat();
    if (mem_type == OpenCLMemType::BUF && mem_type == cur_opencl_op->GetMemType() &&
        in_tensors[i]->GetFormat() == ori_format) {
      continue;
    }
    auto dst_format = (mem_type == OpenCLMemType::IMG) ? in_kernels[i][0]->out_tensors()[0]->GetFormat() : ori_format;
    auto src_format =
      (mem_type == OpenCLMemType::IMG) ? in_tensors[i]->GetFormat() : in_kernels[i][0]->out_tensors()[0]->GetFormat();
    lite::tensor::Tensor *new_tensor = new (std::nothrow) lite::tensor::Tensor();
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new tensor failed!";
      return RET_ERROR;
    }
    new_tensor->CopyTensor(*in_tensors[i]);
    if ((dst_format == schema::Format_NCHW || dst_format == schema::Format_NC4HW4) &&
        (src_format == schema::Format_NHWC || src_format == schema::Format_NHWC4)) {
      auto &shape = new_tensor->shape();
      std::vector<int> dst_shape{shape[0], shape[3], shape[1], shape[2]};
      new_tensor->set_shape(shape);
    }
    if ((dst_format == schema::Format_NHWC || dst_format == schema::Format_NHWC4) &&
        (src_format == schema::Format_NCHW || src_format == schema::Format_NC4HW4)) {
      auto &shape = new_tensor->shape();
      std::vector<int> dst_shape{shape[0], shape[2], shape[3], shape[1]};
      new_tensor->set_shape(shape);
    }
    new_tensor->SetFormat(in_kernels[i][0]->out_tensors()[0]->GetFormat());
    out_tensors->emplace_back(new_tensor);
#ifdef ENABLE_FP16
    KernelKey desc{kGPU, kNumberTypeFloat16, schema::PrimitiveType_ToFormat};
#else
    KernelKey desc{kGPU, kNumberTypeFloat32, schema::PrimitiveType_ToFormat};
#endif
    OpenCLToFormatParameter *parameter = new (std::nothrow) OpenCLToFormatParameter;
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new parameter failed!";
      return RET_ERROR;
    }
    parameter->src_format = src_format;
    parameter->dst_format = dst_format;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op;
    if (mem_type == OpenCLMemType::IMG) {
      in_convert_op =
        lite::GetOpenCLKernel({in_tensors[i]}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter), nullptr, desc);
    } else {
      in_convert_op =
        lite::GetOpenCLKernel({new_tensor}, {in_tensors[i]}, reinterpret_cast<OpParameter *>(parameter), nullptr, desc);
    }
    MS_ASSERT(in_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel create op failed!";
      return RET_ERROR;
    }
    auto in_opencl_op = reinterpret_cast<OpenCLKernel *>(in_convert_op);
    if (mem_type == OpenCLMemType::IMG) {
      for (auto &iv : in_kernels[i]) {
        in_opencl_op->AddOutKernel(iv);
        reinterpret_cast<OpenCLKernel *>(iv)->SetInKernel({in_convert_op});
        reinterpret_cast<OpenCLKernel *>(iv)->set_in_tensors({new_tensor});
      }
    } else {
      for (auto &iv : in_kernels[i]) {
        reinterpret_cast<OpenCLKernel *>(iv)->SetOutKernel({in_convert_op});
        reinterpret_cast<OpenCLKernel *>(iv)->set_out_tensors({new_tensor});
        in_convert_op->AddInKernel(iv);
      }
    }
    out_convert_ops->emplace_back(in_convert_op);
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::Init() {
  allocator_ = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
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

  MallocTensorWithReuse();

  // Map buffer for write, it is not necessary for fine-grained
  for (auto &tensor : in_tensors_) {
    void *data = tensor->Data();
    // It is required with coarse-grained SVM
    if (data != nullptr) {
      data = allocator_->MapBuffer(data, CL_MAP_WRITE, nullptr, true);
      tensor->SetData(data);
    } else {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel input nullptr!";
    }
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::MallocTensorWithReuse() {
  kernel::LiteKernelUtil::InitTensorRefCount(nodes_);
  for (auto *kernel : nodes_) {
    MS_ASSERT(nullptr != kernel);
    kernel::OpenCLKernel *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto &outputs = kernel->out_tensors();
    for (auto i = 0; i < outputs.size(); ++i) {
      auto *output = outputs.at(i);
      MS_ASSERT(nullptr != output);
      if (op_kernel->GetMemType() == OpenCLMemType::IMG) {
        std::vector<size_t> img_size;
        op_kernel->GetImageSize(i, &img_size);
        auto data_ptr = allocator_->Malloc(output->Size(), img_size);
        output->SetData(data_ptr);
      } else {
        output->MallocData(allocator_);
      }
      output->set_allocator(allocator_);
    }
    for (auto input_kernel : kernel->in_kernels()) {
      MS_EXCEPTION_IF_NULL(input_kernel);
      auto ret = input_kernel->DecOutTensorRefCount();
      if (0 != ret) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
      }
    }
  }
  for (auto kernel : out_kernels_) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  for (auto kernel : in_convert_ops_) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  for (auto kernel : out_convert_ops_) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto ret = kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::GetKernelFromToTensor(const std::vector<lite::tensor::Tensor *> &in_tensors,
                                                const std::vector<kernel::LiteKernel *> &in_kernels,
                                                std::vector<std::vector<kernel::LiteKernel *>> *out_kernels,
                                                bool is_from) {
  std::vector<std::set<lite::tensor::Tensor *>> ksets;
  for (auto jv : in_kernels) {
    auto tens = is_from ? jv->in_tensors() : jv->out_tensors();
    std::set<lite::tensor::Tensor *> kset;
    kset.insert(tens.begin(), tens.end());
    ksets.emplace_back(kset);
  }
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    std::vector<kernel::LiteKernel *> kvec;
    for (size_t j = 0; j < in_kernels.size(); ++j) {
      if (ksets[j].count(in_tensors[i])) {
        kvec.emplace_back(in_kernels[j]);
      }
    }
    out_kernels->emplace_back(kvec);
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::UnInit() {
  for (const auto tensor : in_tensors_) {
    if (tensor != nullptr) {
      tensor->FreeData();
    }
  }
  for (const auto tensor : out_tensors_) {
    if (tensor != nullptr) {
      allocator_->UnmapBuffer(tensor->Data());
      tensor->FreeData();
    }
  }
  for (const auto tensor : in_convert_tensors_) {
    if (tensor != nullptr) {
      tensor->FreeData();
      delete tensor;
    }
  }
  for (const auto tensor : out_convert_tensors_) {
    if (tensor != nullptr) {
      tensor->FreeData();
      delete tensor;
    }
  }
  for (const auto parameter : in_parameters_) {
    if (parameter != nullptr) {
      delete parameter;
    }
  }
  for (const auto op : in_convert_ops_) {
    if (op != nullptr) {
      delete op;
    }
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::InferShape() { return RET_OK; }

int SubGraphOpenCLKernel::ReSize() { return RET_OK; }

int SubGraphOpenCLKernel::Run() {
  for (auto &tensor : in_tensors_) {
    allocator_->UnmapBuffer(tensor->Data());
  }

  lite::opencl::OpenCLExecutor executor;
  executor.Run(in_tensors_, out_tensors_, nodes_, allocator_);

  for (auto &tensor : out_tensors_) {
    void *data = allocator_->MapBuffer(tensor->Data(), CL_MAP_READ, nullptr, true);
    tensor->SetData(data);
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
