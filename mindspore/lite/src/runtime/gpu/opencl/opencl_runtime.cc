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

#include "src/runtime/gpu/opencl/opencl_runtime.h"
#include <vector>
#include <numeric>
#include <utility>
#ifdef SHARING_MEM_WITH_OPENGL
#include <EGL/egl.h>
#endif
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/gpu/opencl/opencl_allocator.h"
#include "src/common/file_utils.h"
#ifdef PROGRAM_WITH_IL
#include "src/backend/opencl/cl/program.inc"
#endif

#ifndef ROUND_UP
#define ROUND_UP(x, y) ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y) * static_cast<int>(y))
#endif

using mindspore::kernel::CLErrorCode;

namespace mindspore::lite::opencl {

static std::map<std::string, std::string> g_source_map;
static std::mutex g_mtx;
static std::mutex g_init_mtx;

InitState OpenCLRuntime::init_state_ = UnInit;
OpenCLRuntime *OpenCLRuntime::ocl_runtime_instance_ = nullptr;
size_t OpenCLRuntime::instance_count_ = 0;

OpenCLRuntime *OpenCLRuntime::GetInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  static OpenCLRuntime ocl_runtime;
  if (instance_count_ == 0) {
    ocl_runtime_instance_ = &ocl_runtime;
    ocl_runtime_instance_->Init();
  }
  instance_count_++;
  return ocl_runtime_instance_;
}

void OpenCLRuntime::DeleteInstance() {
  std::unique_lock<std::mutex> lck(g_mtx);
  if (instance_count_ == 0) {
    MS_LOG(ERROR) << "No OpenCLRuntime instance could delete!";
    return;
  }
  instance_count_--;
  if (instance_count_ == 0) {
    ocl_runtime_instance_->Uninit();
  }
}

void printf_callback(const char *buffer, size_t length, size_t final, void *user_data) {
  fwrite(buffer, 1, length, stdout);
}

int OpenCLRuntime::InitGPUDevice(std::vector<cl::Platform> *platforms) {
  MS_ASSERT(platforms);
  // search GPU
  std::vector<cl::Device> devices;
  int ret = RET_OK;
  for (auto &platform : *platforms) {
    std::string platform_name;
    ret = platform.getInfo(CL_PLATFORM_NAME, &platform_name);
    if (ret != CL_SUCCESS) {
      MS_LOG(WARNING) << CLErrorCode(ret);
    }
    ret = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (ret != CL_SUCCESS) {
      MS_LOG(WARNING) << CLErrorCode(ret);
    }
    MS_LOG(INFO) << "Platform (" << platform_name << ") has " << devices.size() << " GPUs";

    if (!devices.empty()) {
      std::string device_name = devices[0].getInfo<CL_DEVICE_NAME>();
      MS_LOG(INFO) << "Find GPU: " << device_name.c_str();
      cl::Platform::setDefault(platform);
      break;
    }
  }

  // not found, return error code.
  if (devices.empty()) {
    MS_LOG(ERROR) << "OpenCL Device not found!";
    return RET_ERROR;
  }

  device_ = new (std::nothrow) cl::Device();
  if (device_ == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL device failed!";
    return RET_ERROR;
  }
  *device_ = devices[0];
  max_work_item_sizes_ = device_->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  max_work_group_size_ = max_work_item_sizes_[0];
  const std::string device_name = device_->getInfo<CL_DEVICE_NAME>();
  const std::string device_version = device_->getInfo<CL_DEVICE_VERSION>();
  const std::string opencl_version = device_->getInfo<CL_DEVICE_OPENCL_C_VERSION>();
  clGetDeviceInfo((*device_)(), CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(cl_uint), &image_pitch_align_, nullptr);
  MS_LOG(INFO) << "Device name:\t" << device_name;
  MS_LOG(INFO) << "Opencl version:\t" << device_version;
  MS_LOG(INFO) << "Image pitch alignment:\t" << image_pitch_align_;
  MS_LOG(INFO) << "Highest OpenCL c version:\t" << opencl_version;
  MS_LOG(INFO) << "Max work item size:\t" << max_work_item_sizes_[0] << " : " << max_work_item_sizes_[1] << " : "
               << max_work_item_sizes_[2];

  gpu_info_ = ParseGpuInfo(device_name, device_version);
  // get cache size, compute units and frequency.
  ret = device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &global_memery_cachesize_);
  if (ret != CL_SUCCESS) {
    MS_LOG(WARNING) << CLErrorCode(ret);
  }
  ret = device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units_);
  if (ret != CL_SUCCESS) {
    MS_LOG(WARNING) << CLErrorCode(ret);
  }
  ret = device_->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_freq_);
  if (ret != CL_SUCCESS) {
    MS_LOG(WARNING) << CLErrorCode(ret);
  }
  cl_device_fp_config fp_config;
  auto success = device_->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fp_config);
  support_fp16_ = CL_SUCCESS == success && fp_config > 0;

  ret = device_->getInfo(CL_DEVICE_SVM_CAPABILITIES, &svm_capabilities_);
  if (ret != CL_SUCCESS || svm_capabilities_ == 0) {
    svm_capabilities_ = 0;
    MS_LOG(INFO) << "SVM capalibilties: "
                 << "NONE";
  } else {
    if (svm_capabilities_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
      MS_LOG(INFO) << "SVM capalibilties: "
                   << "SVM_FINE_GRAIN_BUFFER";
    }
    if (svm_capabilities_ & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      MS_LOG(INFO) << "SVM capalibilties: "
                   << "SVM_COARSE_GRAIN_BUFFER";
    }
    if (svm_capabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
      MS_LOG(INFO) << "SVM capalibilties: "
                   << "SVM_COARSE_GRAIN_SYSTEM";
    }
    if (svm_capabilities_ & CL_DEVICE_SVM_ATOMICS) {
      MS_LOG(INFO) << "SVM capalibilties: "
                   << "SVM_ATOMICS";
    }
  }
  global_memery_size_ = device_->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  max_alloc_size_ = device_->getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  max_image2d_width_ = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  max_image2d_height_ = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  MS_LOG(INFO) << "Address space bits: " << device_->getInfo<CL_DEVICE_ADDRESS_BITS>();
  MS_LOG(INFO) << "Global Mem Size: " << global_memery_size_;
  MS_LOG(INFO) << "Global Mem Cache Size: " << global_memery_cachesize_;
  MS_LOG(INFO) << "Max Alloc Size: " << max_alloc_size_;
  MS_LOG(INFO) << "Compute Unit: " << compute_units_;
  MS_LOG(INFO) << "Clock Frequency: " << max_freq_ << " MHz";
  return RET_OK;
}

int OpenCLRuntime::InitQueue(std::vector<cl::Platform> *platforms) {
  MS_ASSERT(platforms);
  cl_int ret;
#if defined(SHARING_MEM_WITH_OPENGL) && (CL_HPP_TARGET_OPENCL_VERSION >= 120)
  // create context from glcontext
  MS_LOG(INFO) << "Create special opencl context to share with OpenGL";
  cl_context_properties context_prop[] = {CL_GL_CONTEXT_KHR, (cl_context_properties)eglGetCurrentContext(),
                                          CL_EGL_DISPLAY_KHR, (cl_context_properties)eglGetCurrentDisplay(), 0};
  context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, context_prop, nullptr, nullptr, &ret);

  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Create special OpenCL context failed, Create common OpenCL context then.";
    context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, nullptr, nullptr, nullptr, &ret);
    if (context_ == nullptr) {
      delete device_;
      MS_LOG(ERROR) << "Create OpenCL context failed!";
      return RET_ERROR;
    }
  }
#else
  MS_LOG(INFO) << "Create common opencl context";
#ifdef Debug
  std::vector<cl_context_properties> ctx_properties = {CL_CONTEXT_PLATFORM,
                                                       (cl_context_properties)(*platforms)[0](),
                                                       CL_PRINTF_CALLBACK_ARM,
                                                       (cl_context_properties)printf_callback,
                                                       CL_PRINTF_BUFFERSIZE_ARM,
                                                       0x1000000,
                                                       0};
  context_ =
    new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, ctx_properties.data(), nullptr, nullptr, &ret);
#else
  context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, nullptr, nullptr, nullptr, &ret);
#endif
#endif
  if (ret != CL_SUCCESS) {
    delete device_;
    MS_LOG(ERROR) << "Context create failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }

  default_command_queue_ = new (std::nothrow) cl::CommandQueue(*context_, *device_, 0, &ret);
  if (ret != CL_SUCCESS) {
    delete device_;
    delete context_;
    MS_LOG(ERROR) << "Command Queue create failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }

  profiling_command_queue_ = new (std::nothrow) cl::CommandQueue(*context_, *device_, CL_QUEUE_PROFILING_ENABLE, &ret);
  if (ret != CL_SUCCESS) {
    delete device_;
    delete context_;
    delete default_command_queue_;
    MS_LOG(ERROR) << "Profiling command Queue create failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }
  return RET_OK;
}

// Init will get platforms info, get devices info, create opencl context.
int OpenCLRuntime::Init() {
  std::unique_lock<std::mutex> lck(g_init_mtx);
  if (init_state_ == InitSuccess) {
    return RET_OK;
  } else if (init_state_ == InitFailed) {
    return RET_ERROR;
  }
  init_state_ = InitFailed;

  MS_LOG(INFO) << "OpenCL version: CL_TARGET_OPENCL_VERSION " << CL_TARGET_OPENCL_VERSION;
  MS_LOG(INFO) << "CL_HPP_TARGET_OPENCL_VERSION " << CL_HPP_TARGET_OPENCL_VERSION;
  MS_LOG(INFO) << "CL_HPP_MINIMUM_OPENCL_VERSION " << CL_HPP_MINIMUM_OPENCL_VERSION;

#ifdef USE_OPENCL_WRAPPER
  if (!lite::opencl::LoadOpenCLLibrary(&handle_)) {
    MS_LOG(ERROR) << "Load OpenCL symbols failed!";
    return RET_ERROR;
  }
#endif  // USE_OPENCL_WRAPPER
  std::vector<cl::Platform> platforms;
  cl_int ret = cl::Platform::get(&platforms);
  if (platforms.empty()) {
    MS_LOG(ERROR) << "OpenCL Platform not found!" << CLErrorCode(ret);
    return RET_ERROR;
  }
  auto ms_ret = InitGPUDevice(&platforms);
  if (ms_ret != RET_OK) {
    return ms_ret;
  }

  ms_ret = InitQueue(&platforms);
  if (ms_ret != RET_OK) {
    return ms_ret;
  }

  allocator_ = new (std::nothrow) OpenCLAllocator(this);
  if (allocator_ == nullptr) {
    delete device_;
    delete context_;
    delete default_command_queue_;
    delete profiling_command_queue_;
    MS_LOG(ERROR) << "Command OpenCL allocator failed!";
    return RET_ERROR;
  }
#ifdef PROGRAM_WITH_IL
  std::string flag = "";
  binary_program_ = CreateProgramFromIL(g_program_binary, flag);
#endif
  LoadCache();
  init_state_ = InitSuccess;
  MS_LOG(INFO) << "OpenCLRuntime init done!";
  return RET_OK;
}

int OpenCLRuntime::Uninit() {
  std::unique_lock<std::mutex> lck(g_init_mtx);
  if (init_state_ != InitSuccess) {
    return RET_OK;
  }
  StoreCache();
  program_map_.clear();
  delete allocator_;
  delete default_command_queue_;
  delete profiling_command_queue_;
  delete context_;
  delete device_;
  allocator_ = nullptr;
  default_command_queue_ = nullptr;
  profiling_command_queue_ = nullptr;
  context_ = nullptr;
  device_ = nullptr;
  init_state_ = UnInit;
  return RET_OK;
}

OpenCLRuntime::~OpenCLRuntime() { Uninit(); }

cl::Context *OpenCLRuntime::Context() { return context_; }

cl::Device *OpenCLRuntime::Device() { return device_; }

uint64_t OpenCLRuntime::DeviceGlobalMemoryCacheSize() const { return global_memery_cachesize_; }

int OpenCLRuntime::DeviceMaxWorkGroupSize() const { return max_work_group_size_; }

uint32_t OpenCLRuntime::DeviceComputeUnits() const { return compute_units_; }

uint32_t OpenCLRuntime::DeviceMaxFreq() const { return max_freq_; }

// get kernel enqueue max work group size
uint64_t OpenCLRuntime::GetMaxWorkGroupSize(const cl::Kernel &kernel) {
  uint64_t max_workgroup_size = 0;
  int ret = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
  if (ret != CL_SUCCESS) {
    max_workgroup_size = 0;
  }
  return max_workgroup_size;
}

// opencl 2.0 can get SubGroupSize.
uint32_t OpenCLRuntime::GetSubGroupSize(const cl::Kernel &kernel, const cl::NDRange &range) {
  uint32_t sub_group_size = 0;

  if (ADRENO == gpu_info_.type) {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_TARGET_OPENCL_VERSION >= 210 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)
    cl_int cl_ret;
    sub_group_size = kernel.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(*device_, range, &cl_ret);
    if (cl_ret != CL_SUCCESS) {
      CHECK_CL_SUCCESS(cl_ret)
      sub_group_size = 0;
    }
#else
    sub_group_size = 0;
#endif
  }

  return sub_group_size;
}

GpuInfo OpenCLRuntime::GetGpuInfo() { return gpu_info_; }

bool OpenCLRuntime::GetFp16Enable() const { return fp16_enable_; }

// if support fp16, set fp16 will success.
bool OpenCLRuntime::SetFp16Enable(bool enable) {
  fp16_enable_ = enable && support_fp16_;
  return fp16_enable_ == enable;
}

int OpenCLRuntime::BuildKernel(const cl::Kernel &kernel, const std::string &program_name,
                               const std::string &kernel_name, const std::vector<std::string> &build_options_ext,
                               TypeId data_type) {
  std::string build_option = default_build_option_;
  if (fp16_enable_ && data_type != kNumberTypeInt32) {
    build_option +=
      " -DFP16_ENABLE=1 -DFLT=half -DFLT4=half4 -DFLT16=half16 -DAS_FLT4=as_half4 -DAS_UINT4=as_ushort4 -DUINT4=ushort4"
      " -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh -DTO_FLT=convert_half -DTO_FLT4=convert_half4";
  } else {
    build_option +=
      " -DFP16_ENABLE=0 -DFLT=float -DFLT4=float4 -DFLT16=float16 -DAS_FLT4=as_float4 -DAS_UINT4=as_uint4 -DUINT4=uint4"
      " -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef -DTO_FLT=convert_float -DTO_FLT4=convert_float4";
  }
  build_option += " -DMAX_IMAGE2D_WIDTH=" + std::to_string(max_image2d_width_);
  build_option =
    std::accumulate(build_options_ext.begin(), build_options_ext.end(), build_option,
                    [](const std::string &options, const std::string &option) { return options + " " + option; });

  cl::Program program;
  auto program_key = std::make_pair(program_name, build_option);
  auto iter = program_map_.find(program_key);
  if (iter != program_map_.end()) {
    program = iter->second;
  } else {
    flush_cache_ = true;
    auto status = this->LoadProgram(program_name, &program);
    if (!status) {
      MS_LOG(ERROR) << "load program (" << program_name << ") failed!";
      return RET_ERROR;
    }
    status = this->BuildProgram(build_option, program);
    if (!status) {
      MS_LOG(ERROR) << program_name << " build failed!";
      return RET_ERROR;
    }
    program_map_.emplace(program_key, program);
  }

  cl_int ret;
  const_cast<cl::Kernel &>(kernel) = cl::Kernel(program, kernel_name.c_str(), &ret);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << kernel_name << " Kernel create failed:" << CLErrorCode(ret);
    return RET_ERROR;
  }
  return RET_OK;
}

// Run Kernel with 1D, 2D, 3D group size, and local size can be empty.
int OpenCLRuntime::RunKernel(const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local,
                             cl::CommandQueue *command_queue, cl::Event *event) {
  if (command_queue == nullptr) {
    if (profiling_) {
      command_queue = profiling_command_queue_;
    } else {
      command_queue = default_command_queue_;
    }
  }
  MS_ASSERT(local.size() == 0 || local.size() == global.size());
  cl_int ret = CL_SUCCESS;
  ret = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, event);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Kernel execute failed:" << CLErrorCode(ret);
    return RET_ERROR;
  }
  static int cnt = 0;
  const int flush_period = 10;
  if (cnt % flush_period == 0) {
    auto flush_ret = command_queue->flush();
    if (flush_ret != CL_SUCCESS) {
      MS_LOG(WARNING) << "CL Flush failed:" << CLErrorCode(ret);
    }
  }
  cnt++;
  MS_LOG(DEBUG) << "RunKernel success!";
  if (profiling_) {
    event->wait();
  }
  return RET_OK;
}

// get gpu divce type
GpuInfo OpenCLRuntime::ParseGpuInfo(std::string device_name, std::string device_version) {
  GpuInfo info;

  if (device_name == "QUALCOMM Adreno(TM)") {
    info.type = ADRENO;
    sscanf(device_version.c_str(), "%*s%f%*s%d", &info.opencl_version, &info.model_num);

  } else if (device_name.find("Mali") != std::string::npos) {
    info.type = MALI;

    // Mali type MALI-G or MALI_T
    if (device_name.find("Mali-G") != std::string::npos) {
      info.type = MALI_G;
      sscanf(device_name.c_str(), "Mali-G%d", &info.model_num);
    } else if (device_name.find("Mali-T") != std::string::npos) {
      info.type = MALI_T;
      sscanf(device_name.c_str(), "Mali-T%d", &info.model_num);
    }
    sscanf(device_version.c_str(), "%*s%f%*s", &info.opencl_version);
  }

  return info;
}

bool OpenCLRuntime::LoadSource(const std::string &program_name, const std::string &source) {
  auto it_source = g_source_map.find(program_name);
  if (it_source == g_source_map.end()) {
    g_source_map.emplace(program_name, source);
  }
  return true;
}

// load program with program name.
bool OpenCLRuntime::LoadProgram(const std::string &program_name, cl::Program *program) {
  auto it_source = g_source_map.find(program_name);
  if (it_source != g_source_map.end()) {
    cl::Program::Sources sources;
    sources.push_back(it_source->second);
    *program = cl::Program(*context_, sources);
    return true;
  } else {
    MS_LOG(ERROR) << "Can't find kernel source !";
    return false;
  }
}

// build program with build options
bool OpenCLRuntime::BuildProgram(const std::string &build_option, const cl::Program &program) {
  cl_int ret = program.build({*device_}, build_option.c_str());
  if (ret != CL_SUCCESS) {
    if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device_) == CL_BUILD_ERROR) {
      std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device_);
      MS_LOG(ERROR) << "Program build log: " << build_log;
    }
    MS_LOG(ERROR) << "Build program failed: " << CLErrorCode(ret);
    return false;
  }
  return true;
}

int OpenCLRuntime::ReadOrWriteImage(void *buffer, void *data, bool is_read) {
  cl::CommandQueue *command_queue = profiling_ ? profiling_command_queue_ : default_command_queue_;
  auto *image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(buffer));
  if (image == nullptr) {
    MS_LOG(WARNING) << "Can't get Image2D for " << buffer;
    return RET_ERROR;
  }
  ImageSize img_size;
  int ret = allocator_->GetImageSize(buffer, &img_size);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "Can't get GetImageSize for " << buffer;
    return RET_ERROR;
  }
  cl::array<size_t, 3> origin = {0, 0, 0};
  cl::array<size_t, 3> region = {img_size.width, img_size.height, 1};
  if (is_read) {
    ret = command_queue->enqueueReadImage(*image, true, origin, region, 0, 0, data, nullptr, nullptr);
  } else {
    ret = command_queue->enqueueWriteImage(*image, true, origin, region, 0, 0, data, nullptr, nullptr);
  }
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << CLErrorCode(ret);
    return RET_ERROR;
  }
  return RET_OK;
}

int OpenCLRuntime::ReadImage(void *buffer, void *dst_data) { return ReadOrWriteImage(buffer, dst_data, true); }

int OpenCLRuntime::WriteImage(void *buffer, void *src_data) { return ReadOrWriteImage(buffer, src_data, false); }

bool OpenCLRuntime::CopyDeviceMemToHost(void *dst, const void *src, size_t size, cl::CommandQueue *command_queue,
                                        bool sync) const {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  cl_int cl_ret = CL_SUCCESS;
  const cl::Buffer *buffer = static_cast<const cl::Buffer *>(src);
  if (command_queue != nullptr) {
    cl_ret = command_queue->enqueueReadBuffer(*buffer, sync, 0, size, dst);
  }
  return cl_ret == CL_SUCCESS;
}

bool OpenCLRuntime::CopyHostMemToDevice(const void *dst, const void *src, size_t size, cl::CommandQueue *command_queue,
                                        bool sync) const {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  cl_int cl_ret = CL_SUCCESS;
  const cl::Buffer *buffer = static_cast<const cl::Buffer *>(dst);
  if (command_queue != nullptr) {
    cl_ret = command_queue->enqueueWriteBuffer(*buffer, sync, 0, size, src);
  }
  return cl_ret == CL_SUCCESS;
}

void *OpenCLRuntime::MapBuffer(const cl::Buffer &buffer, int flags, size_t size, cl::CommandQueue *command_queue,
                               bool sync) const {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  return command_queue->enqueueMapBuffer(buffer, sync, flags, 0, size);
}

int OpenCLRuntime::MapBuffer(void *host_ptr, int flags, size_t size, cl::CommandQueue *command_queue, bool sync) const {
  if (GetSVMCapabilities() & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
    return RET_OK;
  }
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  return clEnqueueSVMMap(command_queue->get(), sync, flags, host_ptr, size, 0, nullptr, nullptr);
}

void *OpenCLRuntime::MapBuffer(const cl::Image2D &buffer, bool sync, int flags, const std::vector<size_t> &region,
                               cl::CommandQueue *command_queue) const {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  cl::size_type row_pitch;
  cl::size_type slice_pitch;
  cl::array<cl::size_type, 3> origin_{0, 0, 0};
  cl::array<cl::size_type, 3> region_{region[0], region[1], region[2]};
  return command_queue->enqueueMapImage(buffer, sync, flags, origin_, region_, &row_pitch, &slice_pitch);
}

int OpenCLRuntime::UnmapBuffer(const cl::Memory &buffer, void *host_ptr, cl::CommandQueue *command_queue) const {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  return command_queue->enqueueUnmapMemObject(buffer, host_ptr);
}

int OpenCLRuntime::UnmapBuffer(void *host_ptr, cl::CommandQueue *command_queue) const {
  if (GetSVMCapabilities() & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
    return RET_OK;
  }
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  return clEnqueueSVMUnmap(command_queue->get(), host_ptr, 0, nullptr, nullptr);
}

bool OpenCLRuntime::SyncCommandQueue(cl::CommandQueue *command_queue) {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  cl_int ret = command_queue->finish();
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Command queue sync failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }
  return ret == CL_SUCCESS;
}

int OpenCLRuntime::GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id) {
  size_t max_work_group_size;
  cl_int ret = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                        &max_work_group_size, nullptr);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get info CL_KERNEL_WORK_GROUP_SIZE " << CLErrorCode(ret);
  }
  return static_cast<int>(max_work_group_size);
}

cl::Kernel OpenCLRuntime::GetKernelFromBinary(const std::string &kernel_name) {
  cl_int ret = CL_SUCCESS;
  cl::Kernel kernel = cl::Kernel(binary_program_, kernel_name.c_str(), &ret);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Create kernel with binary program failed: " << CLErrorCode(ret);
  }
  return kernel;
}

// build program with IL
cl::Program OpenCLRuntime::CreateProgramFromIL(const std::vector<char> &binary, const std::string &flag) {
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
  cl::Program program = cl::Program(*context_, binary);
  bool status = BuildProgram(default_build_opts_, program);
  if (!status) {
    MS_LOG(ERROR) << "Build program with IL failed!";
  }
  return program;
#else
  MS_LOG(ERROR) << "Create program with IL failed! The compute capabitity of device should be 2.1 and higher.";
  return cl::Program();
#endif
}

// build program with binary
cl::Program OpenCLRuntime::CreateProgramFromBinary(const std::vector<unsigned char> &binary,
                                                   const std::string &build_option) {
  cl::Program program = cl::Program(*context_, {*device_}, {binary});
  bool status = BuildProgram(build_option, program);
  if (!status) {
    MS_LOG(ERROR) << "Build program with binary failed!";
  }
  return program;
}

std::vector<unsigned char> OpenCLRuntime::GetProgramBinary(const cl::Program &program) {
  cl_int ret = CL_SUCCESS;
  auto binarys = program.getInfo<CL_PROGRAM_BINARIES>(&ret);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Get program binary failed: " << CLErrorCode(ret);
  }
  if (binarys.empty()) {
    MS_LOG(ERROR) << "binarys is empty";
    return {};
  }
  return binarys.front();
}

void OpenCLRuntime::LoadCache() {
  if (!enable_cache_) {
    return;
  }
  size_t len;
  std::unique_ptr<char[]> buf(lite::ReadFile(cache_path_.c_str(), &len));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "Load opencl cache fail: buf == nullptr";
    return;
  }
  auto gpu_cache = schema::GetGpuCache(buf.get());
  if (gpu_cache == nullptr) {
    MS_LOG(ERROR) << "Load opencl cache fail: gpu_cache == nullptr";
    return;
  }
  auto *bins = gpu_cache->allBins();
  if (bins == nullptr) {
    MS_LOG(ERROR) << "Load opencl cache fail: bins == nullptr";
    return;
  }
  for (auto i = 0; i < bins->size(); ++i) {
    auto *bin = bins->template GetAs<schema::ProgramBinary>(i);
    if (bin == nullptr) {
      MS_LOG(ERROR) << "kernel_bin[" << i << "] null";
      return;
    }
    auto *pdata = bin->data();
    MS_ASSERT(pdata);
    if (pdata->size() == 0) {
      continue;
    }
    std::vector<unsigned char> binary(pdata->begin(), pdata->end());
    auto program = CreateProgramFromBinary(binary, bin->build_option()->str());
    program_map_.emplace(std::make_pair(bin->program_name()->str(), bin->build_option()->str()), program);
    MS_LOG(INFO) << "LoadCache " << bin->program_name() << " success, size=" << binary.size();
  }
  MS_LOG(INFO) << "Init opencl cache success";
}

void OpenCLRuntime::StoreCache() {
  if (!enable_cache_) {
    return;
  }
  if (!flush_cache_) {
    return;
  }
  auto fbb = std::make_unique<flatbuffers::FlatBufferBuilder>();
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "new opencl FlatBufferBuilder fail";
    return;
  }
  std::vector<flatbuffers::Offset<schema::ProgramBinary>> program_binarys;
  for (const auto &kv : program_map_) {
    auto program_name = kv.first.first;
    auto build_option = kv.first.second;
    cl::Program program = kv.second;
    auto binary = this->GetProgramBinary(program);
    std::vector<int32_t> shape;
    auto tune = schema::CreateTuneParam(*fbb, fbb->CreateVector<int32_t>(shape), fbb->CreateVector<int32_t>(shape),
                                        fbb->CreateVector<int32_t>(shape), fbb->CreateVector<int32_t>(shape));
    auto program_binary = schema::CreateProgramBinary(
      *fbb, fbb->CreateString(program_name), fbb->CreateString(build_option), tune, fbb->CreateVector<uint8_t>(binary));
    program_binarys.emplace_back(program_binary);
    MS_LOG(INFO) << "StoreCache " << program_name << " success, size=" << binary.size();
  }

  auto data = fbb->CreateVector<flatbuffers::Offset<schema::ProgramBinary>>(program_binarys);
  auto name = fbb->CreateString("OpenCLCache");
  auto version = fbb->CreateString(cache_version_);
  auto gpu_cache = schema::CreateGpuCache(*fbb, name, version, data);
  fbb->Finish(gpu_cache);
  uint8_t *buf = fbb->GetBufferPointer();
  lite::WriteToBin(cache_path_, reinterpret_cast<void *>(buf), fbb->GetSize());
  MS_LOG(INFO) << "store opencl cache ok, size=" << fbb->GetSize();
}

}  // namespace mindspore::lite::opencl
