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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_OPENCL_RUNTIME_WRAPPER_H
#define MINDSPORE_LITE_INCLUDE_REGISTRY_OPENCL_RUNTIME_WRAPPER_H
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <type_traits>
#include "CL/cl2.hpp"
#include "include/api/allocator.h"
#include "include/api/status.h"

namespace mindspore::registry::opencl {
class OpenCLRuntimeWrapper {
 public:
  OpenCLRuntimeWrapper() = default;
  ~OpenCLRuntimeWrapper() = default;

  /// \brief Load the OpenCl source code and bind the program name.
  ///
  /// \param[in] program_name Define OpenCl source program name.
  /// \param[in] source Define OpenCl source.
  ///
  /// \return Status as a status identification of loading code.
  Status LoadSource(const std::string &program_name, const std::string &source);

  /// \brief Building OpenCL code.
  ///
  /// \param[in] kernel Used to return the compiled kernel
  /// \param[in] program_name Define OpenCl source program name.
  /// \param[in] kernel_name Define OpenCl source kernel name.
  /// \param[in] build_options_ext Define OpenCl kernel build options.
  ///
  /// \return Status as a status identification of build Kernel
  Status BuildKernel(cl::Kernel *kernel, const std::string &program_name, const std::string &kernel_name,
                     const std::vector<std::string> &build_options_ext = {});

  /// \brief Set kernel argument
  ///
  /// \param[in] kernel Define OpenCl kernel.
  /// \param[in] index Define OpenCl kernel argument index.
  /// \param[in] value Define OpenCl kernel argument value pointer.
  /// \param[in] mem_type Define OpenCl kernel argument value memory type.
  ///
  /// \return Status as a status identification of set kernel argument
  Status SetKernelArg(const cl::Kernel &kernel, uint32_t index, void *const value);

  /// \brief Set kernel argument
  ///
  /// \param[in] kernel Define OpenCl kernel.
  /// \param[in] index Define OpenCl kernel argument index.
  /// \param[in] value Define OpenCl kernel argument value.
  /// \param[in] mem_type Define OpenCl kernel argument value memory type.
  ///
  /// \return Status as a status identification of set kernel argument
  template <typename T>
  typename std::enable_if<!std::is_pointer<T>::value, Status>::type SetKernelArg(const cl::Kernel &kernel,
                                                                                 uint32_t index, const T value) {
    if (const_cast<cl::Kernel &>(kernel).setArg(index, value) != CL_SUCCESS) {
      return kLiteError;
    } else {
      return kSuccess;
    }
  }

  /// \brief Run OpenCl kernel
  ///
  /// \param[in] kernel Define OpenCl kernel.
  /// \param[in] global Define the number of work items
  /// \param[in] local Define the number of work_items in a work_group
  /// \param[in] command_queue Define the command queue
  /// \param[in] event Define event of kernel run
  ///
  /// \return Status as a status identification of run OpenCl kernel
  Status RunKernel(const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local,
                   cl::CommandQueue *command_queue = nullptr, cl::Event *event = nullptr);

  /// \brief Synchronization command queue
  ///
  /// \return Status as a status identification of synchronization command queue
  Status SyncCommandQueue();

  void *MapBuffer(void *host_ptr, int flags, bool sync = true);

  Status UnmapBuffer(void *host_ptr);

  Status ReadImage(void *buffer, void *dst_data);

  Status WriteImage(void *buffer, void *src_data);

  std::shared_ptr<Allocator> GetAllocator();

  uint64_t DeviceMaxWorkGroupSize();

  uint64_t GetMaxImage2DWidth();

  uint64_t GetMaxImage2DHeight();

  uint64_t GetImagePitchAlignment();
};
}  // namespace mindspore::registry::opencl
#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_OPENCL_RUNTIME_WRAPPER_H
