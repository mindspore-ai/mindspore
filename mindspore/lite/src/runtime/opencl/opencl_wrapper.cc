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

#ifdef USE_OPENCL_WRAPPER

#include "src/runtime/opencl/opencl_wrapper.h"
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "utils/log_adapter.h"
#include "src/runtime/opencl/opencl_runtime.h"

namespace mindspore::lite::opencl {

// default opencl library path
static const std::vector<std::string> g_opencl_library_paths = {
#if defined(__APPLE__) || defined(__MACOSX)
  "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"
#elif defined(__ANDROID__)
#if defined(__aarch64__)
  // Mali
  "/system/vendor/lib64/egl/libGLES_mali.so",
  "/system/lib64/egl/libGLES_mali.so",
  // Qualcomm Adreno
  "/system/vendor/lib64/libOpenCL.so",
  "/system/lib64/libOpenCL.so",
#else
  // Qualcomm Adreno
  "/system/vendor/lib/libOpenCL.so", "/system/lib/libOpenCL.so",
  // Mali
  "/system/vendor/lib/egl/libGLES_mali.so", "/system/lib/egl/libGLES_mali.so",
  // other
  "/system/vendor/lib/libPVROCL.so", "/data/data/org.pocl.libs/files/lib/libpocl.so"
#endif
  "libOpenCL.so",
  "libGLES_mali.so",
  "libmali.so",
#elif defined(__linux__)
  "/usr/lib/libOpenCL.so",
  "/usr/local/lib/libOpenCL.so",
  "/usr/local/lib/libpocl.so",
  "/usr/lib64/libOpenCL.so",
  "/usr/lib32/libOpenCL.so",
  "libOpenCL.so",
  // intel
  "/opt/intel/system_studio_2020/opencl/SDK/lib64/libOpenCL.so",
#endif
};

OpenCLWrapper *OpenCLWrapper::GetInstance() {
  static std::once_flag opencl_wrapper_once;
  std::call_once(opencl_wrapper_once,
                 []() { opencl_wrapper_singleton_ = std::shared_ptr<OpenCLWrapper>(new OpenCLWrapper()); });

  return opencl_wrapper_singleton_.get();
}

OpenCLWrapper::OpenCLWrapper() {}

OpenCLWrapper::~OpenCLWrapper() {
  if (nullptr == opencl_wrapper_singleton_.get()) return;
  opencl_wrapper_singleton_->UnLoadOpenCLLibrary();
}

// load default library path
bool OpenCLWrapper::LoadOpenCLLibrary() {
  if (handle_ != nullptr) {
    return true;
  }
  for (const auto &lib_path : g_opencl_library_paths) {
    if (LoadLibraryFromPath(lib_path)) {
      MS_LOG(DEBUG) << "Find a OpenCL dynamic library : " << lib_path;
      return true;
    }
  }
  return false;
}

bool OpenCLWrapper::UnLoadOpenCLLibrary() {
  if (handle_ != nullptr) {
    if (dlclose(handle_) != 0) {
      return false;
    }
    handle_ = nullptr;
    return true;
  }
  return true;
}

bool OpenCLWrapper::LoadLibraryFromPath(const std::string &library_path) {
  handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr) {
    return false;
  }

// load function ptr use dlopen and dlsym.
#define LOAD_OPENCL_FUNCTION_PTR(func_name)                                                    \
  func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name));                   \
  if (func_name == nullptr) {                                                                  \
    MS_LOG(ERROR) << "load func (" << #func_name << ") from (" << library_path << ") failed!"; \
    return false;                                                                              \
  }

  LOAD_OPENCL_FUNCTION_PTR(clGetPlatformIDs);
  LOAD_OPENCL_FUNCTION_PTR(clGetPlatformInfo);
  LOAD_OPENCL_FUNCTION_PTR(clBuildProgram);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueNDRangeKernel);
  LOAD_OPENCL_FUNCTION_PTR(clSetKernelArg);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseKernel);
  LOAD_OPENCL_FUNCTION_PTR(clCreateProgramWithSource);
  LOAD_OPENCL_FUNCTION_PTR(clCreateBuffer);
  LOAD_OPENCL_FUNCTION_PTR(clCreateImage2D);
  LOAD_OPENCL_FUNCTION_PTR(clCreateImage3D);
  LOAD_OPENCL_FUNCTION_PTR(clRetainKernel);
  LOAD_OPENCL_FUNCTION_PTR(clCreateKernel);
  LOAD_OPENCL_FUNCTION_PTR(clGetProgramInfo);
  LOAD_OPENCL_FUNCTION_PTR(clFlush);
  LOAD_OPENCL_FUNCTION_PTR(clFinish);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseProgram);
  LOAD_OPENCL_FUNCTION_PTR(clRetainContext);
  LOAD_OPENCL_FUNCTION_PTR(clGetContextInfo);
  LOAD_OPENCL_FUNCTION_PTR(clCreateProgramWithBinary);
  LOAD_OPENCL_FUNCTION_PTR(clCreateCommandQueue);
  LOAD_OPENCL_FUNCTION_PTR(clGetCommandQueueInfo);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseCommandQueue);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueMapBuffer);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueMapImage);
  LOAD_OPENCL_FUNCTION_PTR(clRetainProgram);
  LOAD_OPENCL_FUNCTION_PTR(clGetProgramBuildInfo);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueReadBuffer);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueWriteBuffer);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueReadImage);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueWriteImage);
  LOAD_OPENCL_FUNCTION_PTR(clWaitForEvents);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseEvent);
  LOAD_OPENCL_FUNCTION_PTR(clCreateContext);
  LOAD_OPENCL_FUNCTION_PTR(clCreateContextFromType);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseContext);
  LOAD_OPENCL_FUNCTION_PTR(clRetainCommandQueue);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueUnmapMemObject);
  LOAD_OPENCL_FUNCTION_PTR(clRetainMemObject);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseMemObject);
  LOAD_OPENCL_FUNCTION_PTR(clGetDeviceInfo);
  LOAD_OPENCL_FUNCTION_PTR(clGetDeviceIDs);
  LOAD_OPENCL_FUNCTION_PTR(clRetainEvent);
  LOAD_OPENCL_FUNCTION_PTR(clGetKernelWorkGroupInfo);
  LOAD_OPENCL_FUNCTION_PTR(clGetEventInfo);
  LOAD_OPENCL_FUNCTION_PTR(clGetEventProfilingInfo);
  LOAD_OPENCL_FUNCTION_PTR(clGetImageInfo);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueCopyImage);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueCopyBufferToImage);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueCopyImageToBuffer);
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
  LOAD_OPENCL_FUNCTION_PTR(clRetainDevice);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseDevice);
  LOAD_OPENCL_FUNCTION_PTR(clCreateImage);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  // LOAD_OPENCL_FUNCTION_PTR(clGetKernelSubGroupInfoKHR);
  LOAD_OPENCL_FUNCTION_PTR(clCreateCommandQueueWithProperties);
  LOAD_OPENCL_FUNCTION_PTR(clGetExtensionFunctionAddress);
  LOAD_OPENCL_FUNCTION_PTR(clSVMAlloc);
  LOAD_OPENCL_FUNCTION_PTR(clSVMFree);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueSVMMap);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueSVMUnmap);
  LOAD_OPENCL_FUNCTION_PTR(clSetKernelArgSVMPointer);
#ifdef PROGRAM_WITH_IL
  LOAD_OPENCL_FUNCTION_PTR(clCreateProgramWithIL);
#endif
#endif

#undef LOAD_OPENCL_FUNCTION_PTR

  return true;
}

}  // namespace mindspore::lite::opencl

// clGetPlatformIDs wrapper, use OpenCLWrapper function. use OpenCLWrapper function.
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetPlatformIDs;
  MS_ASSERT(func != nullptr);
  return func(num_entries, platforms, num_platforms);
}

// clGetPlatformInfo wrapper, use OpenCLWrapper function. use OpenCLWrapper function.
cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetPlatformInfo;
  MS_ASSERT(func != nullptr);
  return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetDeviceIDs wrapper, use OpenCLWrapper function.
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetDeviceIDs;
  MS_ASSERT(func != nullptr);
  return func(platform, device_type, num_entries, devices, num_devices);
}

// clGetDeviceInfo wrapper, use OpenCLWrapper function.
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetDeviceInfo;
  MS_ASSERT(func != nullptr);
  return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clCreateContext wrapper, use OpenCLWrapper function.
cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateContext;
  MS_ASSERT(func != nullptr);
  return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

// clCreateContextFromType wrapper, use OpenCLWrapper function.
cl_context clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateContextFromType;
  MS_ASSERT(func != nullptr);
  return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

// clRetainContext wrapper, use OpenCLWrapper function.
cl_int clRetainContext(cl_context context) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainContext;
  MS_ASSERT(func != nullptr);
  return func(context);
}

// clReleaseContext wrapper, use OpenCLWrapper function.
cl_int clReleaseContext(cl_context context) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseContext;
  MS_ASSERT(func != nullptr);
  return func(context);
}

// clGetContextInfo wrapper, use OpenCLWrapper function.
cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetContextInfo;
  MS_ASSERT(func != nullptr);
  return func(context, param_name, param_value_size, param_value, param_value_size_ret);
}

// clCreateProgramWithSource wrapper, use OpenCLWrapper function.
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateProgramWithSource;
  MS_ASSERT(func != nullptr);
  return func(context, count, strings, lengths, errcode_ret);
}

// clGetProgramInfo wrapper, use OpenCLWrapper function.
cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetProgramInfo;
  MS_ASSERT(func != nullptr);
  return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetProgramBuildInfo wrapper, use OpenCLWrapper function.
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetProgramBuildInfo;
  MS_ASSERT(func != nullptr);
  return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clRetainProgram wrapper, use OpenCLWrapper function.
cl_int clRetainProgram(cl_program program) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainProgram;
  MS_ASSERT(func != nullptr);
  return func(program);
}

// clReleaseProgram wrapper, use OpenCLWrapper function.
cl_int clReleaseProgram(cl_program program) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseProgram;
  MS_ASSERT(func != nullptr);
  return func(program);
}

// clBuildProgram wrapper, use OpenCLWrapper function.
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clBuildProgram;
  MS_ASSERT(func != nullptr);
  return func(program, num_devices, device_list, options, pfn_notify, user_data);
}

// clCreateKernel wrapper, use OpenCLWrapper function.
cl_kernel clCreateKernel(cl_program program, const char *kernelName, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateKernel;
  MS_ASSERT(func != nullptr);
  return func(program, kernelName, errcode_ret);
}

// clRetainKernel wrapper, use OpenCLWrapper function.
cl_int clRetainKernel(cl_kernel kernel) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainKernel;
  MS_ASSERT(func != nullptr);
  return func(kernel);
}

// clReleaseKernel wrapper, use OpenCLWrapper function.
cl_int clReleaseKernel(cl_kernel kernel) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseKernel;
  MS_ASSERT(func != nullptr);
  return func(kernel);
}

// clSetKernelArg wrapper, use OpenCLWrapper function.
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clSetKernelArg;
  MS_ASSERT(func != nullptr);
  return func(kernel, arg_index, arg_size, arg_value);
}

// clCreateBuffer wrapper, use OpenCLWrapper function.
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateBuffer;
  MS_ASSERT(func != nullptr);
  return func(context, flags, size, host_ptr, errcode_ret);
}

// clRetainMemObject wrapper, use OpenCLWrapper function.
cl_int clRetainMemObject(cl_mem memobj) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainMemObject;
  MS_ASSERT(func != nullptr);
  return func(memobj);
}

// clReleaseMemObject wrapper, use OpenCLWrapper function.
cl_int clReleaseMemObject(cl_mem memobj) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseMemObject;
  MS_ASSERT(func != nullptr);
  return func(memobj);
}

// clGetImageInfo wrapper, use OpenCLWrapper function.
cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetImageInfo;
  MS_ASSERT(func != nullptr);
  return func(image, param_name, param_value_size, param_value, param_value_size_ret);
}

// clRetainCommandQueue wrapper, use OpenCLWrapper function.
cl_int clRetainCommandQueue(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clReleaseCommandQueue wrapper, use OpenCLWrapper function.
cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clEnqueueReadBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueReadBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueWriteBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueWriteBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list,
              event);
}

// clEnqueueWriteImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin,
                           const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr,
                           cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueWriteImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr,
              num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueReadImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t *origin,
                          const size_t *region, size_t row_pitch, size_t slice_pitch, void *ptr,
                          cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueReadImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list,
              event_wait_list, event);
}

// clEnqueueMapBuffer wrapper, use OpenCLWrapper function.
void *clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                         size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                         cl_event *event, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueMapBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list,
              event, errcode_ret);
}

// clEnqueueMapImage wrapper, use OpenCLWrapper function.
void *clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueMapImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch,
              num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

// clEnqueueUnmapMemObject wrapper, use OpenCLWrapper function.
cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueUnmapMemObject;
  MS_ASSERT(func != nullptr);
  return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

// clGetKernelWorkGroupInfo wrapper, use OpenCLWrapper function.
cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetKernelWorkGroupInfo;
  MS_ASSERT(func != nullptr);
  return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetEventProfilingInfo wrapper, use OpenCLWrapper function.
cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetEventProfilingInfo;
  MS_ASSERT(func != nullptr);
  return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// clEnqueueNDRangeKernel wrapper, use OpenCLWrapper function.
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueNDRangeKernel;
  MS_ASSERT(func != nullptr);
  return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
              num_events_in_wait_list, event_wait_list, event);
}

// clWaitForEvents wrapper, use OpenCLWrapper function.
cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clWaitForEvents;
  MS_ASSERT(func != nullptr);
  return func(num_events, event_list);
}

// clRetainEvent wrapper, use OpenCLWrapper function.
cl_int clRetainEvent(cl_event event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainEvent;
  MS_ASSERT(func != nullptr);
  return func(event);
}

// clReleaseEvent wrapper, use OpenCLWrapper function.
cl_int clReleaseEvent(cl_event event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseEvent;
  MS_ASSERT(func != nullptr);
  return func(event);
}

// clGetEventInfo wrapper, use OpenCLWrapper function.
cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetEventInfo;
  MS_ASSERT(func != nullptr);
  return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// clFlush wrapper, use OpenCLWrapper function.
cl_int clFlush(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clFlush;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clFinish wrapper, use OpenCLWrapper function.
cl_int clFinish(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clFinish;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clCreateImage2D wrapper, use OpenCLWrapper function.
cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateImage2D;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, imageWidth, imageHeight, image_row_pitch, host_ptr, errcode_ret);
}

// clCreateImage3D wrapper, use OpenCLWrapper function.
cl_mem clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t imageDepth, size_t image_row_pitch, size_t image_slice_pitch,
                       void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateImage3D;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, imageWidth, imageHeight, imageDepth, image_row_pitch, image_slice_pitch,
              host_ptr, errcode_ret);
}

// clCreateCommandQueue wrapper, use OpenCLWrapper function.
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(context, device, properties, errcode_ret);
}

// clGetCommandQueueInfo wrapper, use OpenCLWrapper function.
cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size,
                             void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetCommandQueueInfo;
  MS_ASSERT(func != nullptr);
  return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

// clEnqueueCopyImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyImage(cl_command_queue queue, cl_mem src_image, cl_mem dst_image, const size_t *src_origin,
                          const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueCopyImage;
  MS_ASSERT(func != nullptr);
  return func(queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list,
              event);
}

// clEnqueueCopyBufferToImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
                                  size_t src_offset, const size_t *dst_origin, const size_t *region,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueCopyBufferToImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list,
              event_wait_list, event);
}

// clEnqueueCopyImageToBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
                                  const size_t *src_origin, const size_t *region, size_t dst_offset,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueCopyImageToBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list,
              event_wait_list, event);
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

// clRetainDevice wrapper, use OpenCLWrapper function.
cl_int clRetainDevice(cl_device_id device) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clRetainDevice;
  MS_ASSERT(func != nullptr);
  return func(device);
}

// clReleaseDevice wrapper, use OpenCLWrapper function.
cl_int clReleaseDevice(cl_device_id device) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clReleaseDevice;
  MS_ASSERT(func != nullptr);
  return func(device);
}

// clCreateImage wrapper, use OpenCLWrapper function.
cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateImage;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
#if 0
// clGetKernelSubGroupInfoKHR wrapper, use OpenCLWrapper function.
cl_int clGetKernelSubGroupInfoKHR(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name,
                                  size_t input_value_size, const void *input_value, size_t param_value_size,
                                  void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetKernelSubGroupInfoKHR;
  MS_ASSERT(func != nullptr);
  return func(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value,
              param_value_size_ret);
}
#endif

// clCreateCommandQueueWithProperties wrapper, use OpenCLWrapper function.
cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                                    const cl_queue_properties *properties, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateCommandQueueWithProperties;
  MS_ASSERT(func != nullptr);
  return func(context, device, properties, errcode_ret);
}

// clGetExtensionFunctionAddress wrapper, use OpenCLWrapper function.
void *clGetExtensionFunctionAddress(const char *func_name) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clGetExtensionFunctionAddress;
  MS_ASSERT(func != nullptr);
  return func(func_name);
}
// clCreateProgramWithIL wrapper, use OpenCLWrapper function.
cl_program clCreateProgramWithIL(cl_context context, const void *il, size_t length, cl_int *ret) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clCreateProgramWithIL;
  MS_ASSERT(func != nullptr);
  return func(context, il, length, ret);
}

// clSVMAlloc wrapper, use OpenCLWrapper function.
void *clSVMAlloc(cl_context context, cl_mem_flags flags, size_t size, cl_uint align) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clSVMAlloc;
  MS_ASSERT(func != nullptr);
  return func(context, flags, size, align);
}

// clSVMFree wrapper, use OpenCLWrapper function.
void clSVMFree(cl_context context, void *buffer) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clSVMFree;
  MS_ASSERT(func != nullptr);
  func(context, buffer);
}

// clEnqueueSVMMap wrapper, use OpenCLWrapper function.
cl_int clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking, cl_map_flags flags, void *host_ptr,
                       size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueSVMMap;
  MS_ASSERT(func != nullptr);
  return func(command_queue, blocking, flags, host_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueSVMUnmap wrapper, use OpenCLWrapper function.
cl_int clEnqueueSVMUnmap(cl_command_queue command_queue, void *host_ptr, cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clEnqueueSVMUnmap;
  MS_ASSERT(func != nullptr);
  return func(command_queue, host_ptr, num_events_in_wait_list, event_wait_list, event);
}

// clSetKernelArgSVMPointer wrapper, use OpenCLWrapper function.
cl_int clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint index, const void *host_ptr) {
  auto func = mindspore::lite::opencl::OpenCLWrapper::GetInstance()->clSetKernelArgSVMPointer;
  MS_ASSERT(func != nullptr);
  return func(kernel, index, host_ptr);
}
#endif

#endif  // USE_OPENCL_WRAPPER

