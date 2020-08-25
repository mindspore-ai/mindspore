/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
j* you may not use this file except in compliance with the License.
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

#ifndef MINDSPORE_LITE_SRC_OPENCL_RUNTIME_H_
#define MINDSPORE_LITE_SRC_OPENCL_RUNTIME_H_

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include "utils/log_adapter.h"
#include "src/runtime/opencl/opencl_wrapper.h"
#include "src/runtime/opencl/opencl_allocator.h"

namespace mindspore::lite::opencl {

enum GpuType { OTHER = 0, ADRENO = 1, MALI = 2, MALI_T = 3, MALI_G = 4 };

struct GpuInfo {
  GpuType type = OTHER;
  int model_num = 0;
  float opencl_version = 0;
};

class OpenCLRuntime {
 public:
  static OpenCLRuntime *GetInstance();
  static void DeleteInstance();

  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime &) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

  int Init();

  cl::Context *Context();
  cl::Device *Device();
  OpenCLAllocator *GetAllocator() { return allocator_; }
  cl::CommandQueue *GetDefaultCommandQueue() { return default_command_queue_; }
  uint64_t DeviceGlobalMemoryCacheSize() const;
  int DeviceMaxWorkGroupSize() const;
  uint32_t DeviceComputeUnits() const;
  uint32_t DeviceMaxFreq() const;
  uint64_t GetMaxWorkGroupSize(const cl::Kernel &kernel);
  uint32_t GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range = cl::NullRange);
  GpuInfo GetGpuInfo();
  bool GetFp16Enable() const;
  bool SetFp16Enable(bool enable);
  bool GetSVMEnable() const { return svm_enable_; }
  void SetSVMEnable(bool enable) { svm_enable_ = enable; }
  const std::vector<size_t> &GetWorkItemSize() const { return max_work_item_sizes_; }
  uint32_t GetImagePitchAlignment() const { return image_pitch_align_; }
  cl_device_svm_capabilities GetSVMCapabilities() const { return svm_enable_ ? svm_capabilities_ : 0; }

  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value, cl_int>::type SetKernelArg(cl::Kernel &kernel, uint32_t index,
                                                                                const T value,
                                                                                const MemType mem_type = MemType::IMG) {
    switch (mem_type) {
      case MemType::SVM: {
        MS_LOG(DEBUG) << "Set kernel arg[" << index << "] SVM pointer " << value;
        return kernel.setArg(index, value);
      }
      case MemType::BUF: {
        cl::Buffer *buffer = reinterpret_cast<cl::Buffer *>(allocator_->GetBuffer(value));
        MS_LOG(DEBUG) << "Set kernel arg[" << index << "] OpenCL Buffer " << buffer << ", host_ptr: " << value;
        return kernel.setArg(index, *buffer);
      }
      case MemType::IMG: {
        cl::Image2D *image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(value));
        if (image == nullptr) {
          MS_LOG(WARNING) << "Can't get Image2D, try to use Buffer. Please confirm the buffer type.";
          cl::Buffer *buffer = reinterpret_cast<cl::Buffer *>(allocator_->GetBuffer(value));
          MS_LOG(DEBUG) << "Set kernel arg[" << index << "] OpenCL Buffer " << buffer << ", host_ptr: " << value;
          return kernel.setArg(index, *buffer);
        }
        MS_LOG(DEBUG) << "Set kernel arg[" << index << "] OpenCL Image2D " << image << ", host_ptr: " << value;
        return kernel.setArg(index, *image);
      }
      default:
        MS_LOG(ERROR) << "Unsupport opencl memory type: " << static_cast<int>(mem_type);
    }
  }

  template <typename T>
  typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type SetKernelArg(
    cl::Kernel &kernel, uint32_t index, const T value, const MemType mem_type = MemType::IMG) {
    return kernel.setArg(index, value);
  }

  cl::Program CreateProgramFromIL(const std::vector<char> &binary, const std::string &flag);
  cl::Program CreateProgramFromBinary(const std::vector<std::vector<unsigned char>> &binary, const std::string &flag);
  cl::Kernel GetKernelFromBinary(const std::string &kernel_name);
  std::vector<std::vector<unsigned char>> GetProgramBinaries(const cl::Program &program);
  bool LoadSource(const std::string &program_name, const std::string &source);
  int BuildKernel(cl::Kernel &kernel, const std::string &program_name, const std::string &kernel_name,
                  const std::set<std::string> &build_options);
  int RunKernel(const cl::Kernel &kernel, const std::vector<size_t> &global, const std::vector<size_t> &local,
                cl::CommandQueue *command_queue);
  bool CopyDeviceMemToHost(void *dst, const void *src, size_t size, cl::CommandQueue *command_queue = nullptr,
                           bool sync = false) const;
  bool CopyHostMemToDevice(const void *dst, const void *src, size_t size, cl::CommandQueue *command_queue = nullptr,
                           bool sync = false) const;
  void *MapBuffer(const cl::Buffer &buffer, int map_flags, size_t size, cl::CommandQueue *command_queue = nullptr,
                  bool sync = false) const;
  void *MapBuffer(const cl::Image2D &buffer, bool sync, int flags, const std::vector<size_t> &region,
                  cl::CommandQueue *command_queue = nullptr) const;
  int MapBuffer(void *host_ptr, int map_flags, size_t size, cl::CommandQueue *command_queue = nullptr,
                bool sync = false) const;
  int UnmapBuffer(const cl::Memory &buffer, void *host_ptr, cl::CommandQueue *command_queue = nullptr) const;
  int UnmapBuffer(void *host_ptr, cl::CommandQueue *command_queue = nullptr) const;
  bool SyncCommandQueue(cl::CommandQueue *command_queue = nullptr);

  /**
   * Get kernel max worker group size.
   * @param kernel
   * @param device_id
   * @return max_work_group_size
   */
  int GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id);

 private:
  OpenCLRuntime();
  GpuInfo ParseGpuInfo(std::string device_name, std::string device_version);

  bool LoadProgram(const std::string &program_name, cl::Program *program);
  bool BuildProgram(const std::string &build_options, const cl::Program &program);

 private:
  static bool init_done_;
  cl::CommandQueue *default_command_queue_{nullptr};
  cl::Context *context_{nullptr};
  cl::Device *device_{nullptr};
  OpenCLAllocator *allocator_{nullptr};
  std::map<std::string, cl::Program> program_map_;
  cl::Program binary_program_{0};
  uint64_t global_memery_cachesize_{0};
  int max_work_group_size;
  uint32_t compute_units_{0};
  uint32_t max_freq_{0};
  std::string default_build_opts_{""};
  GpuInfo gpu_info_;
  bool support_fp16_{false};
  bool fp16_enable_{false};
  bool svm_enable_{false};
  cl_device_svm_capabilities svm_capabilities_{0};
  cl_uint image_pitch_align_{0};
  std::vector<size_t> max_work_item_sizes_;
};

}  // namespace mindspore::lite::opencl
#endif  // MINDSPORE_LITE_SRC_OPENCL_RUNTIME_H_
