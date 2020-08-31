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

#include "src/runtime/opencl/opencl_runtime.h"
#include <vector>
#include <numeric>
#ifdef SHARING_MEM_WITH_OPENGL
#include <EGL/egl.h>
#endif
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/opencl/opencl_allocator.h"
#ifdef PROGRAM_WITH_IL
#include "src/backend/opencl/cl/program.inc"
#endif

#ifndef ROUND_UP
#define ROUND_UP(x, y) ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y) * static_cast<int>(y))
#endif

using mindspore::kernel::CLErrorCode;

namespace mindspore::lite::opencl {

static std::map<std::string, std::string> g_opencl_program_map;
static std::mutex g_mtx;
static std::mutex g_init_mtx;

bool OpenCLRuntime::init_done_ = false;
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
  }
  instance_count_--;
  if (instance_count_ == 0) {
    ocl_runtime_instance_->Uninit();
  }
}

OpenCLRuntime::OpenCLRuntime() { default_build_opts_ = " -cl-mad-enable -cl-fast-relaxed-math -Werror"; }

void printf_callback(const char *buffer, size_t length, size_t final, void *user_data) {
  fwrite(buffer, 1, length, stdout);
}

// Init will get platforms info, get devices info, create opencl context.
int OpenCLRuntime::Init() {
  std::unique_lock<std::mutex> lck(g_init_mtx);

  if (init_done_) {
    return RET_OK;
  }
  MS_LOG(INFO) << "OpenCL version: CL_TARGET_OPENCL_VERSION " << CL_TARGET_OPENCL_VERSION;
  MS_LOG(INFO) << "CL_HPP_TARGET_OPENCL_VERSION " << CL_HPP_TARGET_OPENCL_VERSION;
  MS_LOG(INFO) << "CL_HPP_MINIMUM_OPENCL_VERSION " << CL_HPP_MINIMUM_OPENCL_VERSION;

#ifdef USE_OPENCL_WRAPPER
  if (OpenCLWrapper::GetInstance()->LoadOpenCLLibrary() == false) {
    MS_LOG(ERROR) << "Load OpenCL symbols failed!";
    return RET_ERROR;
  }
#endif  // USE_OPENCL_WRAPPER

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    MS_LOG(ERROR) << "OpenCL Platform not found!";
    return RET_ERROR;
  }

  // search GPU
  std::vector<cl::Device> devices;
  for (auto it = platforms.begin(); it != platforms.end(); ++it) {
    std::string platform_name;
    it->getInfo(CL_PLATFORM_NAME, &platform_name);
    it->getDevices(CL_DEVICE_TYPE_GPU, &devices);
    MS_LOG(INFO) << "Platform (" << platform_name << ") has " << devices.size() << " GPUs";

    if (devices.size() > 0) {
      std::string device_name = devices[0].getInfo<CL_DEVICE_NAME>();
      MS_LOG(INFO) << "Find GPU: " << device_name.c_str();
      cl::Platform::setDefault(*it);
      break;
    }
  }

  // not found, return error code.
  if (devices.size() == 0) {
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
  cl_int ret;
#if defined(SHARING_MEM_WITH_OPENGL) && (CL_HPP_TARGET_OPENCL_VERSION >= 120)
  // create context from glcontext
  MS_LOG(INFO) << "Create special opencl context to share with OpenGL";
  cl_context_properties context_prop[] = {CL_GL_CONTEXT_KHR, (cl_context_properties)eglGetCurrentContext(),
                                          CL_EGL_DISPLAY_KHR, (cl_context_properties)eglGetCurrentDisplay(), 0};
  context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, context_prop, nullptr, nullptr, &ret);

  if (ret != CL_SUCCESS || context_ == nullptr) {
    MS_LOG(ERROR) << "Create special OpenCL context failed, Create common OpenCL context then.";
    context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, nullptr, nullptr, nullptr, &ret);
    if (context_ == nullptr) {
      MS_LOG(ERROR) << "Create OpenCL context failed!";
      return RET_ERROR;
    }
  }
#else
  MS_LOG(INFO) << "Create common opencl context";
  context_ = new (std::nothrow) cl::Context(std::vector<cl::Device>{*device_}, nullptr, nullptr, nullptr, &ret);
#endif
  if (ret != CL_SUCCESS || context_ == nullptr) {
    MS_LOG(ERROR) << "Context create failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }

  // get cache size, compute units and frequency.
  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &global_memery_cachesize_);
  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units_);
  device_->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_freq_);
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

  MS_LOG(INFO) << "Global Mem Cache Size: " << global_memery_cachesize_;
  MS_LOG(INFO) << "Compute Unit: " << compute_units_;
  MS_LOG(INFO) << "Clock Frequency: " << max_freq_ << " MHz";

  const cl_command_queue_properties properties = 0;
#if MS_OPENCL_PROFILE
  properties |= CL_QUEUE_PROFILING_ENABLE;
#endif

  default_command_queue_ = new (std::nothrow) cl::CommandQueue(*context_, *device_, properties, &ret);
  if (ret != CL_SUCCESS || default_command_queue_ == nullptr) {
    MS_LOG(ERROR) << "Command Queue create failed: " << CLErrorCode(ret);
    return RET_ERROR;
  }

  allocator_ = new (std::nothrow) OpenCLAllocator(this);
  if (allocator_ == nullptr) {
    MS_LOG(ERROR) << "Command OpenCL allocator failed!";
    return RET_ERROR;
  }
#ifdef PROGRAM_WITH_IL
  std::string flag = "";
  binary_program_ = CreateProgramFromIL(g_program_binary, flag);
#endif
  init_done_ = true;
  MS_LOG(INFO) << "OpenCLRuntime init done!";

  return RET_OK;
}

int OpenCLRuntime::Uninit() {
  program_map_.clear();
  delete allocator_;
  delete default_command_queue_;
  delete context_;
  delete device_;
  allocator_ = nullptr;
  default_command_queue_ = nullptr;
  context_ = nullptr;
  device_ = nullptr;
#ifdef USE_OPENCL_WRAPPER
  OpenCLWrapper::GetInstance()->UnLoadOpenCLLibrary();
#endif
  init_done_ = false;
  return RET_OK;
}

OpenCLRuntime::~OpenCLRuntime() { Uninit(); }

cl::Context *OpenCLRuntime::Context() { return context_; }

cl::Device *OpenCLRuntime::Device() { return device_; }

uint64_t OpenCLRuntime::DeviceGlobalMemoryCacheSize() const { return global_memery_cachesize_; }

int OpenCLRuntime::DeviceMaxWorkGroupSize() const { return max_work_group_size; }

uint32_t OpenCLRuntime::DeviceComputeUnits() const { return compute_units_; }

uint32_t OpenCLRuntime::DeviceMaxFreq() const { return max_freq_; }

// get kernel enqueue max work group size
uint64_t OpenCLRuntime::GetMaxWorkGroupSize(const cl::Kernel &kernel) {
  uint64_t max_workgroup_size = 0;
  int ret = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &max_workgroup_size);
  if (ret != 0) max_workgroup_size = 0;
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

int OpenCLRuntime::BuildKernel(cl::Kernel &kernel, const std::string &program_name, const std::string &kernel_name,
                               const std::set<std::string> &build_options) {
  std::string build_options_str;
  // set default macro
  if (fp16_enable_) {
    // fp16 enable, kernel will use half and read_imageh and write_imageh.
    build_options_str =
      "-DFLT=half -DFLT4=half4 -DFLT16=half16 "
      "-DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh -DTO_FLT4=convert_half4 ";
  } else {
    // fp16 not enable, kernel will use float and read_imagef and write_imagef.
    build_options_str =
      "-DFLT=float -DFLT4=float4 -DFLT16=float16 "
      "-DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef -DTO_FLT4=convert_float4 ";
  }

  auto build_options_ext = std::accumulate(
    build_options.begin(), build_options.end(), std::string(""),
    [](const std::string &options, const std::string &option) -> std::string {
       auto res = options + " " + option;
       return res;
       });
  build_options_str += default_build_opts_;
  // program identifier = program_name + build_options
  std::string build_program_key = program_name + build_options_str + build_options_ext;

  auto build_program_it = program_map_.find(build_program_key);
  cl::Program program;
  // if search program identifier exist, then use it.
  if (build_program_it != program_map_.end()) {
    program = build_program_it->second;
  } else {
    // load program and build program
    auto status = this->LoadProgram(program_name, &program);
    if (!status) {
      MS_LOG(ERROR) << "load program (" << program_name << ") failed!";
      return RET_ERROR;
    }
    status = this->BuildProgram(build_options_str, program);
    if (!status) {
      MS_LOG(ERROR) << program_name << " build failed!";
      return RET_ERROR;
    }
    program_map_.emplace(build_program_key, program);
  }

  cl_int ret;
  kernel = cl::Kernel(program, kernel_name.c_str(), &ret);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << kernel_name << " Kernel create failed:" << CLErrorCode(ret);
    return RET_ERROR;
  }
  return RET_OK;
}

// Run Kernel with 1D, 2D, 3D group size, and local size can be empty.
int OpenCLRuntime::RunKernel(const cl::Kernel &kernel, const std::vector<size_t> &global,
                             const std::vector<size_t> &local, cl::CommandQueue *command_queue) {
  if (command_queue == nullptr) {
    command_queue = default_command_queue_;
  }
  MS_ASSERT(local.size() == 0 || local.size() == global.size());
  std::vector<size_t> internal_global_ws = global;
  for (size_t i = 0; i < local.size(); ++i) {
    internal_global_ws[i] = ROUND_UP(global[i], local[i]);
  }

  MS_LOG(DEBUG) << "global size: " << global.size() << ", local size: " << local.size();
  for (size_t i = 0; i < global.size(); i++) {
    MS_LOG(DEBUG) << "global[" << i << "] = " << global[i];
  }
  for (size_t i = 0; i < local.size(); i++) {
    MS_LOG(DEBUG) << "local[" << i << "] = " << local[i];
  }

  cl::NDRange global_range = cl::NullRange;
  cl::NDRange local_range = cl::NullRange;
  if (global.size() == 1) {
    global_range = cl::NDRange(internal_global_ws[0]);
    if (!local.empty()) {
      local_range = cl::NDRange(local[0]);
    }
  } else if (global.size() == 2) {
    global_range = cl::NDRange(internal_global_ws[0], internal_global_ws[1]);
    if (!local.empty()) {
      local_range = cl::NDRange(local[0], local[1]);
    }
  } else if (global.size() == 3) {
    global_range = cl::NDRange(internal_global_ws[0], internal_global_ws[1], internal_global_ws[2]);
    if (!local.empty()) {
      local_range = cl::NDRange(local[0], local[1], local[2]);
    }
  } else {
    MS_LOG(ERROR) << "Not supported NDRange!";
    return RET_ERROR;
  }

  cl::Event event;
  cl_int ret = CL_SUCCESS;
  ret = command_queue->enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range, nullptr, &event);

  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Kernel execute failed:" << CLErrorCode(ret);
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "RunKernel success!";
#if MS_OPENCL_PROFILE
  event.wait();
  cl_ulong time_start;
  cl_ulong time_end;
  event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
  event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
  double nanoSeconds = time_end - time_start;
  MS_LOG(INFO) << "OpenCl Execution time is: " << nanoSeconds / 1000000.0 << "ms";
#endif
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
  auto it_source = g_opencl_program_map.find(program_name);
  if (it_source == g_opencl_program_map.end()) {
    g_opencl_program_map.emplace(program_name, source);
  }
  return true;
}

// load program with program name.
bool OpenCLRuntime::LoadProgram(const std::string &program_name, cl::Program *program) {
  auto it_source = g_opencl_program_map.find(program_name);
  if (it_source != g_opencl_program_map.end()) {
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
bool OpenCLRuntime::BuildProgram(const std::string &build_options, const cl::Program &program) {
  cl_int ret = program.build({*device_}, build_options.c_str());
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
  return command_queue->enqueueMapSVM(host_ptr, sync, flags, size);
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
  return command_queue->enqueueUnmapSVM(host_ptr);
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
cl::Program OpenCLRuntime::CreateProgramFromBinary(const std::vector<std::vector<unsigned char>> &binary,
                                                   const std::string &flag) {
  cl::Program program = cl::Program(*context_, {*device_}, binary);
  bool status = BuildProgram(default_build_opts_, program);
  if (!status) {
    MS_LOG(ERROR) << "Build program with binary failed!";
  }
  return program;
}

std::vector<std::vector<unsigned char>> OpenCLRuntime::GetProgramBinaries(const cl::Program &program) {
  cl_int ret = CL_SUCCESS;
  auto binary = program.getInfo<CL_PROGRAM_BINARIES>(&ret);
  if (ret != CL_SUCCESS) {
    MS_LOG(ERROR) << "Get program binary failed: " << CLErrorCode(ret);
  }
  return binary;
}
}  // namespace mindspore::lite::opencl
