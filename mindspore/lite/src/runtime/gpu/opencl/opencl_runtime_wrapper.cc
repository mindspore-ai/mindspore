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

#include "include/registry/opencl_runtime_wrapper.h"
#include <dlfcn.h>
#ifdef SHARING_MEM_WITH_OPENGL
#include <EGL/egl.h>
#endif
#include <vector>
#include <numeric>
#include <utility>
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/gpu/opencl/opencl_allocator.h"
#include "src/common/file_utils.h"
#include "src/runtime/gpu/opencl/opencl_runtime.h"

using mindspore::kernel::CLErrorCode;

namespace mindspore::registry::opencl {
Status OpenCLRuntimeWrapper::LoadSource(const std::string &program_name, const std::string &source) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  const std::string program_name_ext = "provider_" + program_name;
  if (ocl_runtime->LoadSource(program_name_ext, source)) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

Status OpenCLRuntimeWrapper::BuildKernel(cl::Kernel *kernel, const std::string &program_name,
                                         const std::string &kernel_name,
                                         const std::vector<std::string> &build_options_ext) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  const std::string program_name_ext = "provider_" + program_name;
  if (ocl_runtime->BuildKernel(*kernel, program_name_ext, kernel_name, build_options_ext, false) == RET_OK) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

Status OpenCLRuntimeWrapper::SetKernelArg(const cl::Kernel &kernel, uint32_t index, void *const value) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->SetKernelArg(kernel, index, value) != CL_SUCCESS) {
    return kLiteError;
  } else {
    return kSuccess;
  }
}

Status OpenCLRuntimeWrapper::RunKernel(const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local,
                                       cl::CommandQueue *command_queue, cl::Event *event) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->RunKernel(kernel, global, local, command_queue, event) == RET_OK) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

Status OpenCLRuntimeWrapper::SyncCommandQueue() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->SyncCommandQueue()) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

void *OpenCLRuntimeWrapper::MapBuffer(void *host_ptr, int flags, bool sync) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->GetAllocator()->MapBuffer(host_ptr, flags, nullptr, sync);
}

Status OpenCLRuntimeWrapper::UnmapBuffer(void *host_ptr) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->GetAllocator()->UnmapBuffer(host_ptr, nullptr) == RET_OK) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

Status OpenCLRuntimeWrapper::ReadImage(void *buffer, void *dst_data) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->ReadImage(buffer, dst_data) == RET_OK) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

Status OpenCLRuntimeWrapper::WriteImage(void *buffer, void *src_data) {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  if (ocl_runtime->WriteImage(buffer, src_data) == RET_OK) {
    return kSuccess;
  } else {
    return kLiteError;
  }
}

std::shared_ptr<Allocator> OpenCLRuntimeWrapper::GetAllocator() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->GetAllocator();
}

uint64_t OpenCLRuntimeWrapper::DeviceMaxWorkGroupSize() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->DeviceMaxWorkGroupSize();
}

uint64_t OpenCLRuntimeWrapper::GetMaxImage2DWidth() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->GetMaxImage2DWidth();
}

uint64_t OpenCLRuntimeWrapper::GetMaxImage2DHeight() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->GetMaxImage2DHeight();
}

uint64_t OpenCLRuntimeWrapper::GetImagePitchAlignment() {
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap;
  lite::opencl::OpenCLRuntime *ocl_runtime = ocl_runtime_wrap.GetInstance();
  return ocl_runtime->GetImagePitchAlignment();
}
}  // namespace mindspore::registry::opencl
