/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_RUNTIME_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_RUNTIME_H_
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <type_traits>
#include "dtype/type_id.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/gpu/opencl/opencl_wrapper.h"
#include "src/litert/kernel/gpu/opencl/opencl_allocator.h"
#include "schema/gpu_cache_generated.h"
#define EXT_ARM_IMPORT_MEMORY_HOST "cl_arm_import_memory_host"

namespace mindspore::lite::opencl {
enum GpuType { OTHER = 0, ADRENO = 1, MALI = 2, MALI_T = 3, MALI_G = 4, MALI_G78 = 5 };
enum TuningMode { DEFAULT = 0, FAST = 1, EXTREME = 2 };
enum InitState { UnInit = 0, InitSuccess = 1, InitFailed = 2 };

struct GpuInfo {
  GpuType type = OTHER;
};
class OpenCLRuntimeInnerWrapper;
class OpenCLRuntimeWrapper;
class OpenCLRuntime {
 public:
  friend OpenCLRuntimeInnerWrapper;
  friend OpenCLRuntimeWrapper;
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime &) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

  int Init();
  int Uninit();

  cl::Context *Context();
  cl::Device *Device();
  std::shared_ptr<OpenCLAllocator> GetAllocator() { return allocator_; }
  cl::CommandQueue *GetDefaultCommandQueue() { return profiling_ ? profiling_command_queue_ : default_command_queue_; }
  uint64_t DeviceGlobalMemoryCacheSize() const;
  uint64_t DeviceMaxWorkGroupSize() const;
  uint32_t DeviceComputeUnits() const;
  uint32_t DeviceMaxFreq() const;
  uint64_t GetMaxWorkGroupSize(const cl::Kernel &kernel);
  uint32_t GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range = cl::NullRange);
  uint64_t GetGlobalMemSize() { return global_memery_size_; }
  uint64_t GetMaxAllocSize() { return max_alloc_size_; }
  uint64_t GetMaxImage2DWidth() { return max_image2d_width_; }
  uint64_t GetMaxImage2DHeight() { return max_image2d_height_; }
  GpuInfo GetGpuInfo();
  bool GetFp16Enable() const;
  bool SetFp16Enable(bool enable);
  bool GetGLTextureEnable() const;
  bool SetGLTextureEnable(bool enable);

  void SetGLContext(void *gl_context) { gl_context_ = gl_context; }
  void *GetGLContext() const { return gl_context_; }
  bool CheckGLContext() const { return (GetGLContext() != nullptr); }

  void SetGLDisplay(void *gl_display) { gl_display_ = gl_display; }
  void *GetGLDisplay() const { return gl_display_; }
  bool CheckGLDisplay() const { return (GetGLDisplay() != nullptr); }

  bool GetSVMEnable() const { return svm_enable_; }
  void SetSVMEnable(bool enable) { svm_enable_ = enable; }
  const std::vector<size_t> &GetWorkItemSize() const { return max_work_item_sizes_; }
  uint32_t GetImagePitchAlignment() const { return image_pitch_align_; }
  cl_device_svm_capabilities GetSVMCapabilities() const { return svm_enable_ ? svm_capabilities_ : 0; }

  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value, cl_int>::type SetKernelArg(const cl::Kernel &kernel,
                                                                                uint32_t index, const T value,
                                                                                bool force_buffer = false) {
    if (value == nullptr) {
      MS_LOG(ERROR) << "value is nullptr.";
      return CL_INVALID_VALUE;
    }
    auto svm_capabilities = GetSVMCapabilities();
    if (svm_capabilities) {
      MS_LOG(DEBUG) << "Set kernel arg[" << index << "] SVM pointer " << value;
      return clSetKernelArgSVMPointer(kernel.get(), index, value);
    }
    lite::opencl::MemType mem_type;
    void *buffer = allocator_->GetOpenclMemPtr(value, &mem_type, force_buffer);
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "buffer is nullptr.";
      return CL_INVALID_VALUE;
    }
    MS_LOG(DEBUG) << "Set kernel arg[" << index << "] OpenCL "
                  << (mem_type == lite::opencl::MemType::IMG ? "Image " : "Buffer ") << buffer
                  << ", host_ptr: " << value;
    if (mem_type == lite::opencl::MemType::IMG) {
      return const_cast<cl::Kernel &>(kernel).setArg(index, *reinterpret_cast<cl::Image2D *>(buffer));
    } else {
      return const_cast<cl::Kernel &>(kernel).setArg(index, *reinterpret_cast<cl::Buffer *>(buffer));
    }
  }

  template <typename T>
  typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type SetKernelArg(const cl::Kernel &kernel,
                                                                                 uint32_t index, const T value) {
    return const_cast<cl::Kernel &>(kernel).setArg(index, value);
  }

  cl::Program CreateProgramFromIL(const std::vector<char> &binary, const std::string &flag);
  cl::Program CreateProgramFromBinary(const std::vector<unsigned char> &binary, const std::string &flag);
  cl::Kernel GetKernelFromBinary(const std::string &kernel_name);
  std::vector<unsigned char> GetProgramBinary(const cl::Program &program);
  bool LoadSource(const std::string &program_name, const std::string &source);
  int BuildKernel(const cl::Kernel &kernel, const std::string &program_name, const std::string &kernel_name,
                  const std::vector<std::string> &build_options_ext = {}, const bool is_builtin = true);
  int RunKernel(const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local,
                cl::CommandQueue *command_queue = nullptr, cl::Event *event = nullptr);
  int ReadOrWriteImage(void *buffer, void *data, bool is_read);
  int ReadImage(void *buffer, void *dst_data);
  int WriteImage(void *buffer, void *src_data);
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
  void SetTuningMode(TuningMode mode) { tuning_mode_ = mode; }
  TuningMode GetTuningMode() const { return tuning_mode_; }

  bool isProfiling() const { return profiling_; }
  void SetProfiling(bool profiling) { profiling_ = profiling; }
  bool isExtensionEnable(std::string ext) { return supported_extensions_.find(ext) != std::string::npos; }
  cl::Buffer *CreateSharedMemoryBuffer(size_t size, void *host_ptr);
  uint GetCacheLineSize() const { return cache_line_size_; }

 private:
  static OpenCLRuntime *GetInstance();
  static void DeleteInstance();
  OpenCLRuntime() = default;
  GpuInfo ParseGpuInfo(std::string device_name, std::string device_version);

  bool LoadProgram(const std::string &program_name, cl::Program *program);
  bool BuildProgram(const std::string &build_options, const cl::Program &program);
  int InitGPUDevice(std::vector<cl::Platform> *platforms);
  int InitQueue(std::vector<cl::Platform> *platforms);

 private:
  static InitState init_state_;
  static size_t instance_count_;
  static OpenCLRuntime *ocl_runtime_instance_;
  cl::CommandQueue *default_command_queue_{nullptr};
  cl::CommandQueue *profiling_command_queue_{nullptr};
  cl::Context *context_{nullptr};
  cl::Device *device_{nullptr};
  std::shared_ptr<OpenCLAllocator> allocator_{nullptr};
  std::map<std::pair<std::string, std::string>, cl::Program> program_map_;
  cl::Program binary_program_;
  uint64_t global_memery_cachesize_{0};
  uint64_t global_memery_size_{0};
  uint64_t max_alloc_size_{0};
  uint64_t max_image2d_width_{0};
  uint64_t max_image2d_height_{0};
  uint64_t max_work_group_size_{1};
  uint32_t compute_units_{0};
  uint32_t max_freq_{0};
  std::string default_build_option_{"-cl-mad-enable -cl-fast-relaxed-math -Werror"};
  GpuInfo gpu_info_;
  bool support_fp16_{false};
  bool fp16_enable_{false};
  bool svm_enable_{false};
  cl_device_svm_capabilities svm_capabilities_{0};
  cl_uint image_pitch_align_{0};
  std::vector<size_t> max_work_item_sizes_;
  void *handle_{nullptr};
  bool enable_gl_texture_{false};
  void *gl_context_{nullptr};
  void *gl_display_{nullptr};
  TuningMode tuning_mode_{TuningMode::DEFAULT};
#if MS_OPENCL_PROFILE
  bool profiling_{true};
#else
  bool profiling_{false};
  std::string supported_extensions_{""};
  uint cache_line_size_{1};
#endif
  // for cache
 private:
  void LoadCache();
  int StoreCache();
#ifdef MS_OPENCL_BINARY_CACHE
  bool enable_cache_{true};
#else
  bool enable_cache_{false};
#endif
  bool flush_cache_{false};
  std::string cache_path_{"/data/local/tmp/.opencl_cache"};
  const std::string cache_version_{"V0.1"};
};

class OpenCLRuntimeInnerWrapper {
 public:
  OpenCLRuntimeInnerWrapper() { ocl_runtime_ = OpenCLRuntime::GetInstance(); }
  ~OpenCLRuntimeInnerWrapper() { OpenCLRuntime::DeleteInstance(); }
  OpenCLRuntimeInnerWrapper(const OpenCLRuntimeInnerWrapper &) = delete;
  OpenCLRuntimeInnerWrapper &operator=(const OpenCLRuntimeInnerWrapper &) = delete;
  OpenCLRuntime *GetInstance() { return ocl_runtime_; }

 private:
  OpenCLRuntime *ocl_runtime_{nullptr};
};
}  // namespace mindspore::lite::opencl
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_GPU_OPENCL_OPENCL_RUNTIME_H_
