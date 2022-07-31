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
#include "src/litert/kernel/opencl/kernel/gl_to_cl.h"
#include <map>
#include <string>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/cl/gl_to_cl.cl.inc"

namespace mindspore::kernel {
const cl_GLenum kGlTexture2D = 0x0DE1;

int GLToCLOpenCLKernel::CheckSpecs() { return RET_OK; }

int GLToCLOpenCLKernel::PreProcess() {
  if (this->out_mem_type_ == lite::opencl::MemType::IMG) return OpenCLKernel::PreProcess();
  auto ret = ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  return RET_OK;
}

int GLToCLOpenCLKernel::SetConstArgs() { return RET_OK; }

int GLToCLOpenCLKernel::SetGlobalLocal() {
  global_size_ = {W_ * UP_DIV(C_, C4NUM), N_ * H_};
  local_size_ = {1, 1};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int GLToCLOpenCLKernel::Prepare() {
  auto out_tensor = out_tensors_.front();
  auto in_tensor = in_tensors_.front();
  std::string kernel_name;
  if (out_mem_type_ == lite::opencl::MemType::IMG) {
    kernel_name = "GLTexture2D_to_IMG";
    auto output = GpuTensorInfo(out_tensor);
    N_ = output.N;
    H_ = output.H;
    W_ = output.W;
    C_ = output.C;
  } else {
    kernel_name = "IMG_to_GLTexture2D";
    auto input = GpuTensorInfo(in_tensor);
    N_ = input.N;
    H_ = input.H;
    W_ = input.W;
    C_ = input.C;
  }

  this->set_name(kernel_name);

  const std::string program_name = "GLTexture2D_to_img";
  std::string source = gl_to_cl_source;
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }

  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int GLToCLOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  cl::Context *context = ocl_runtime_->Context();
  auto dst_mem_type = out_mem_type_;
  cl_int status;
  if (dst_mem_type == lite::opencl::MemType::IMG) {
    auto in_tensor = in_tensors_.front();
    auto data_type = in_tensor->data_type();
    if (data_type != kNumberTypeGLUInt) {
      MS_LOG(ERROR) << "Unsupported data type " << data_type;
      return RET_ERROR;
    }
    cl_GLuint *gl_texture_id = reinterpret_cast<cl_GLuint *>(in_tensor->data());
    auto img_gl = cl::ImageGL(*context, CL_MEM_READ_ONLY, kGlTexture2D, 0, *gl_texture_id, &status);
    if (status != CL_SUCCESS) {
      MS_LOG(ERROR) << "Create ImageGL failed : " << status << std::endl;
      return RET_ERROR;
    }
    if (kernel_.setArg(0, sizeof(cl_mem), &img_gl) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data(),
                                   (dst_mem_type == lite::opencl::MemType::BUF)) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
      MS_LOG(ERROR) << "RunKernel failed.";
      return RET_ERROR;
    }
  } else {
    auto out_tensor = out_tensors_.front();
    cl_GLuint *gl_texture_id = reinterpret_cast<cl_GLuint *>(out_tensor->data());
    auto img_gl = cl::ImageGL(*context, CL_MEM_WRITE_ONLY, kGlTexture2D, 0, *gl_texture_id, &status);
    if (status != CL_SUCCESS) {
      MS_LOG(ERROR) << "Create ImageGL failed : " << status << std::endl;
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (kernel_.setArg(1, sizeof(cl_mem), &img_gl) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
      MS_LOG(ERROR) << "RunKernel failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int GLToCLOpenCLKernel::InferShape() {
  if (!InferShapeDone()) {
    out_tensors_.front()->set_shape(in_tensors_.front()->shape());
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
