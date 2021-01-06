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

#include "src/runtime/gpu/opencl/opencl_wrapper.h"
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "src/common/log_adapter.h"

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
  "/system/vendor/lib/libOpenCL.so",
  "/system/lib/libOpenCL.so",
  // Mali
  "/system/vendor/lib/egl/libGLES_mali.so",
  "/system/lib/egl/libGLES_mali.so",
  // other
  "/system/vendor/lib/libPVROCL.so",
  "/data/data/org.pocl.libs/files/lib/libpocl.so"
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

bool UnLoadOpenCLLibrary(void *handle) {
  if (handle != nullptr) {
    if (dlclose(handle) != 0) {
      return false;
    }
    return true;
  }
  return true;
}

bool LoadLibraryFromPath(const std::string &library_path, void **handle_ptr) {
  if (handle_ptr == nullptr) {
    return false;
  }

  *handle_ptr = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle_ptr == nullptr) {
    return false;
  }

// load function ptr use dlopen and dlsym.
#define LOAD_OPENCL_FUNCTION_PTR(func_name)                                                    \
  func_name = reinterpret_cast<func_name##Func>(dlsym(*handle_ptr, #func_name));               \
  if (func_name == nullptr) {                                                                  \
    MS_LOG(ERROR) << "load func (" << #func_name << ") from (" << library_path << ") failed!"; \
    UnLoadOpenCLLibrary(*handle_ptr);                                                          \
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
#if CL_TARGET_OPENCL_VERSION >= 120
  LOAD_OPENCL_FUNCTION_PTR(clRetainDevice);
  LOAD_OPENCL_FUNCTION_PTR(clReleaseDevice);
  LOAD_OPENCL_FUNCTION_PTR(clCreateImage);
  LOAD_OPENCL_FUNCTION_PTR(clEnqueueFillImage);
#endif
#if CL_TARGET_OPENCL_VERSION >= 200
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

  return true;
}
// load default library path
bool LoadOpenCLLibrary(void **handle_ptr) {
  if (handle_ptr == nullptr) {
    return false;
  }
  if (*handle_ptr != nullptr) {
    return true;
  }
  auto it =
    std::find_if(g_opencl_library_paths.begin(), g_opencl_library_paths.end(),
                 [&](const std::string &lib_path) { return lite::opencl::LoadLibraryFromPath(lib_path, handle_ptr); });
  if (it != g_opencl_library_paths.end()) {
    MS_LOG(DEBUG) << "Find a OpenCL dynamic library : " << *it;
    return true;
  }
  return false;
}

#define CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr

CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
CL_DEFINE_FUNC_PTR(clBuildProgram);
CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
CL_DEFINE_FUNC_PTR(clSetKernelArg);
CL_DEFINE_FUNC_PTR(clReleaseKernel);
CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
CL_DEFINE_FUNC_PTR(clCreateBuffer);
CL_DEFINE_FUNC_PTR(clCreateImage2D);
CL_DEFINE_FUNC_PTR(clCreateImage3D);
CL_DEFINE_FUNC_PTR(clRetainKernel);
CL_DEFINE_FUNC_PTR(clCreateKernel);
CL_DEFINE_FUNC_PTR(clGetProgramInfo);
CL_DEFINE_FUNC_PTR(clFlush);
CL_DEFINE_FUNC_PTR(clFinish);
CL_DEFINE_FUNC_PTR(clReleaseProgram);
CL_DEFINE_FUNC_PTR(clRetainContext);
CL_DEFINE_FUNC_PTR(clGetContextInfo);
CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);
CL_DEFINE_FUNC_PTR(clCreateCommandQueue);
CL_DEFINE_FUNC_PTR(clGetCommandQueueInfo);
CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
CL_DEFINE_FUNC_PTR(clEnqueueMapImage);
CL_DEFINE_FUNC_PTR(clEnqueueCopyImage);
CL_DEFINE_FUNC_PTR(clRetainProgram);
CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
CL_DEFINE_FUNC_PTR(clEnqueueWriteImage);
CL_DEFINE_FUNC_PTR(clEnqueueReadImage);
CL_DEFINE_FUNC_PTR(clWaitForEvents);
CL_DEFINE_FUNC_PTR(clReleaseEvent);
CL_DEFINE_FUNC_PTR(clCreateContext);
CL_DEFINE_FUNC_PTR(clCreateContextFromType);
CL_DEFINE_FUNC_PTR(clReleaseContext);
CL_DEFINE_FUNC_PTR(clRetainCommandQueue);
CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
CL_DEFINE_FUNC_PTR(clRetainMemObject);
CL_DEFINE_FUNC_PTR(clReleaseMemObject);
CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
CL_DEFINE_FUNC_PTR(clRetainEvent);
CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
CL_DEFINE_FUNC_PTR(clGetEventInfo);
CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
CL_DEFINE_FUNC_PTR(clGetImageInfo);
CL_DEFINE_FUNC_PTR(clEnqueueCopyBufferToImage);
CL_DEFINE_FUNC_PTR(clEnqueueCopyImageToBuffer);
#if CL_TARGET_OPENCL_VERSION >= 120
CL_DEFINE_FUNC_PTR(clRetainDevice);
CL_DEFINE_FUNC_PTR(clReleaseDevice);
CL_DEFINE_FUNC_PTR(clCreateImage);
CL_DEFINE_FUNC_PTR(clEnqueueFillImage);
#endif
#if CL_TARGET_OPENCL_VERSION >= 200
CL_DEFINE_FUNC_PTR(clGetKernelSubGroupInfoKHR);
CL_DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
CL_DEFINE_FUNC_PTR(clGetExtensionFunctionAddress);
CL_DEFINE_FUNC_PTR(clCreateProgramWithIL);
CL_DEFINE_FUNC_PTR(clSVMAlloc);
CL_DEFINE_FUNC_PTR(clSVMFree);
CL_DEFINE_FUNC_PTR(clEnqueueSVMMap);
CL_DEFINE_FUNC_PTR(clEnqueueSVMUnmap);
CL_DEFINE_FUNC_PTR(clSetKernelArgSVMPointer);
#endif
#undef LOAD_OPENCL_FUNCTION_PTR
}  // namespace mindspore::lite::opencl

// clGetPlatformIDs wrapper, use OpenCLWrapper function. use OpenCLWrapper function.
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  auto func = mindspore::lite::opencl::clGetPlatformIDs;
  MS_ASSERT(func != nullptr);
  return func(num_entries, platforms, num_platforms);
}

// clGetPlatformInfo wrapper, use OpenCLWrapper function. use OpenCLWrapper function.
cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetPlatformInfo;
  MS_ASSERT(func != nullptr);
  return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetDeviceIDs wrapper, use OpenCLWrapper function.
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
  auto func = mindspore::lite::opencl::clGetDeviceIDs;
  MS_ASSERT(func != nullptr);
  return func(platform, device_type, num_entries, devices, num_devices);
}

// clGetDeviceInfo wrapper, use OpenCLWrapper function.
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetDeviceInfo;
  MS_ASSERT(func != nullptr);
  return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clCreateContext wrapper, use OpenCLWrapper function.
cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateContext;
  MS_ASSERT(func != nullptr);
  return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

// clCreateContextFromType wrapper, use OpenCLWrapper function.
cl_context clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateContextFromType;
  MS_ASSERT(func != nullptr);
  return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

// clRetainContext wrapper, use OpenCLWrapper function.
cl_int clRetainContext(cl_context context) {
  auto func = mindspore::lite::opencl::clRetainContext;
  MS_ASSERT(func != nullptr);
  return func(context);
}

// clReleaseContext wrapper, use OpenCLWrapper function.
cl_int clReleaseContext(cl_context context) {
  auto func = mindspore::lite::opencl::clReleaseContext;
  MS_ASSERT(func != nullptr);
  return func(context);
}

// clGetContextInfo wrapper, use OpenCLWrapper function.
cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetContextInfo;
  MS_ASSERT(func != nullptr);
  return func(context, param_name, param_value_size, param_value, param_value_size_ret);
}

// clCreateProgramWithSource wrapper, use OpenCLWrapper function.
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateProgramWithSource;
  MS_ASSERT(func != nullptr);
  return func(context, count, strings, lengths, errcode_ret);
}

// clCreateProgramWithBinary wrapper, use OpenCLWrapper function.
cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *devices_list,
                                     const size_t *lengths, const unsigned char **binaries, cl_int *binary_status,
                                     cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateProgramWithBinary;
  MS_ASSERT(func != nullptr);
  return func(context, num_devices, devices_list, lengths, binaries, binary_status, errcode_ret);
}

// clGetProgramInfo wrapper, use OpenCLWrapper function.
cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetProgramInfo;
  MS_ASSERT(func != nullptr);
  return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetProgramBuildInfo wrapper, use OpenCLWrapper function.
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetProgramBuildInfo;
  MS_ASSERT(func != nullptr);
  return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clRetainProgram wrapper, use OpenCLWrapper function.
cl_int clRetainProgram(cl_program program) {
  auto func = mindspore::lite::opencl::clRetainProgram;
  MS_ASSERT(func != nullptr);
  return func(program);
}

// clReleaseProgram wrapper, use OpenCLWrapper function.
cl_int clReleaseProgram(cl_program program) {
  auto func = mindspore::lite::opencl::clReleaseProgram;
  MS_ASSERT(func != nullptr);
  return func(program);
}

// clBuildProgram wrapper, use OpenCLWrapper function.
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) {
  auto func = mindspore::lite::opencl::clBuildProgram;
  MS_ASSERT(func != nullptr);
  return func(program, num_devices, device_list, options, pfn_notify, user_data);
}

// clCreateKernel wrapper, use OpenCLWrapper function.
cl_kernel clCreateKernel(cl_program program, const char *kernelName, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateKernel;
  MS_ASSERT(func != nullptr);
  return func(program, kernelName, errcode_ret);
}

// clRetainKernel wrapper, use OpenCLWrapper function.
cl_int clRetainKernel(cl_kernel kernel) {
  auto func = mindspore::lite::opencl::clRetainKernel;
  MS_ASSERT(func != nullptr);
  return func(kernel);
}

// clReleaseKernel wrapper, use OpenCLWrapper function.
cl_int clReleaseKernel(cl_kernel kernel) {
  auto func = mindspore::lite::opencl::clReleaseKernel;
  MS_ASSERT(func != nullptr);
  return func(kernel);
}

// clSetKernelArg wrapper, use OpenCLWrapper function.
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  auto func = mindspore::lite::opencl::clSetKernelArg;
  MS_ASSERT(func != nullptr);
  return func(kernel, arg_index, arg_size, arg_value);
}

// clCreateBuffer wrapper, use OpenCLWrapper function.
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateBuffer;
  MS_ASSERT(func != nullptr);
  return func(context, flags, size, host_ptr, errcode_ret);
}

// clRetainMemObject wrapper, use OpenCLWrapper function.
cl_int clRetainMemObject(cl_mem memobj) {
  auto func = mindspore::lite::opencl::clRetainMemObject;
  MS_ASSERT(func != nullptr);
  return func(memobj);
}

// clReleaseMemObject wrapper, use OpenCLWrapper function.
cl_int clReleaseMemObject(cl_mem memobj) {
  auto func = mindspore::lite::opencl::clReleaseMemObject;
  MS_ASSERT(func != nullptr);
  return func(memobj);
}

// clGetImageInfo wrapper, use OpenCLWrapper function.
cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetImageInfo;
  MS_ASSERT(func != nullptr);
  return func(image, param_name, param_value_size, param_value, param_value_size_ret);
}

// clRetainCommandQueue wrapper, use OpenCLWrapper function.
cl_int clRetainCommandQueue(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::clRetainCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clReleaseCommandQueue wrapper, use OpenCLWrapper function.
cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::clReleaseCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clEnqueueReadBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueReadBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueWriteBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueWriteBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list,
              event);
}

// clEnqueueWriteImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin,
                           const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr,
                           cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueWriteImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr,
              num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueReadImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t *origin,
                          const size_t *region, size_t row_pitch, size_t slice_pitch, void *ptr,
                          cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueReadImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list,
              event_wait_list, event);
}

// clEnqueueMapBuffer wrapper, use OpenCLWrapper function.
void *clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                         size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                         cl_event *event, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clEnqueueMapBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list,
              event, errcode_ret);
}

// clEnqueueMapImage wrapper, use OpenCLWrapper function.
void *clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clEnqueueMapImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch,
              num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

// clEnqueueUnmapMemObject wrapper, use OpenCLWrapper function.
cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueUnmapMemObject;
  MS_ASSERT(func != nullptr);
  return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

// clGetKernelWorkGroupInfo wrapper, use OpenCLWrapper function.
cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetKernelWorkGroupInfo;
  MS_ASSERT(func != nullptr);
  return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

// clGetEventProfilingInfo wrapper, use OpenCLWrapper function.
cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetEventProfilingInfo;
  MS_ASSERT(func != nullptr);
  return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// clEnqueueNDRangeKernel wrapper, use OpenCLWrapper function.
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueNDRangeKernel;
  MS_ASSERT(func != nullptr);
  return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
              num_events_in_wait_list, event_wait_list, event);
}

// clWaitForEvents wrapper, use OpenCLWrapper function.
cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  auto func = mindspore::lite::opencl::clWaitForEvents;
  MS_ASSERT(func != nullptr);
  return func(num_events, event_list);
}

// clRetainEvent wrapper, use OpenCLWrapper function.
cl_int clRetainEvent(cl_event event) {
  auto func = mindspore::lite::opencl::clRetainEvent;
  MS_ASSERT(func != nullptr);
  return func(event);
}

// clReleaseEvent wrapper, use OpenCLWrapper function.
cl_int clReleaseEvent(cl_event event) {
  auto func = mindspore::lite::opencl::clReleaseEvent;
  MS_ASSERT(func != nullptr);
  return func(event);
}

// clGetEventInfo wrapper, use OpenCLWrapper function.
cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetEventInfo;
  MS_ASSERT(func != nullptr);
  return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// clFlush wrapper, use OpenCLWrapper function.
cl_int clFlush(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::clFlush;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clFinish wrapper, use OpenCLWrapper function.
cl_int clFinish(cl_command_queue command_queue) {
  auto func = mindspore::lite::opencl::clFinish;
  MS_ASSERT(func != nullptr);
  return func(command_queue);
}

// clCreateImage2D wrapper, use OpenCLWrapper function.
cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateImage2D;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, imageWidth, imageHeight, image_row_pitch, host_ptr, errcode_ret);
}

// clCreateImage3D wrapper, use OpenCLWrapper function.
cl_mem clCreateImage3D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t imageDepth, size_t image_row_pitch, size_t image_slice_pitch,
                       void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateImage3D;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, imageWidth, imageHeight, imageDepth, image_row_pitch, image_slice_pitch,
              host_ptr, errcode_ret);
}

// clCreateCommandQueue wrapper, use OpenCLWrapper function.
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateCommandQueue;
  MS_ASSERT(func != nullptr);
  return func(context, device, properties, errcode_ret);
}

// clGetCommandQueueInfo wrapper, use OpenCLWrapper function.
cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size,
                             void *param_value, size_t *param_value_size_ret) {
  auto func = mindspore::lite::opencl::clGetCommandQueueInfo;
  MS_ASSERT(func != nullptr);
  return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

// clEnqueueCopyImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyImage(cl_command_queue queue, cl_mem src_image, cl_mem dst_image, const size_t *src_origin,
                          const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueCopyImage;
  MS_ASSERT(func != nullptr);
  return func(queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list,
              event);
}

// clEnqueueCopyBufferToImage wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
                                  size_t src_offset, const size_t *dst_origin, const size_t *region,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueCopyBufferToImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list,
              event_wait_list, event);
}

// clEnqueueCopyImageToBuffer wrapper, use OpenCLWrapper function.
cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
                                  const size_t *src_origin, const size_t *region, size_t dst_offset,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueCopyImageToBuffer;
  MS_ASSERT(func != nullptr);
  return func(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list,
              event_wait_list, event);
}

#if CL_TARGET_OPENCL_VERSION >= 120

// clRetainDevice wrapper, use OpenCLWrapper function.
cl_int clRetainDevice(cl_device_id device) {
  auto func = mindspore::lite::opencl::clRetainDevice;
  MS_ASSERT(func != nullptr);
  return func(device);
}

// clReleaseDevice wrapper, use OpenCLWrapper function.
cl_int clReleaseDevice(cl_device_id device) {
  auto func = mindspore::lite::opencl::clReleaseDevice;
  MS_ASSERT(func != nullptr);
  return func(device);
}

// clCreateImage wrapper, use OpenCLWrapper function.
cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateImage;
  MS_ASSERT(func != nullptr);
  return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_int clEnqueueFillImage(cl_command_queue command_queue, cl_mem image, const void *fill_color, const size_t *origin,
                          const size_t *region, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                          cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueFillImage;
  MS_ASSERT(func != nullptr);
  return func(command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event);
}

#endif

#if CL_TARGET_OPENCL_VERSION >= 200

// clCreateCommandQueueWithProperties wrapper, use OpenCLWrapper function.
cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                                    const cl_queue_properties *properties, cl_int *errcode_ret) {
  auto func = mindspore::lite::opencl::clCreateCommandQueueWithProperties;
  MS_ASSERT(func != nullptr);
  return func(context, device, properties, errcode_ret);
}

// clGetExtensionFunctionAddress wrapper, use OpenCLWrapper function.
void *clGetExtensionFunctionAddress(const char *func_name) {
  auto func = mindspore::lite::opencl::clGetExtensionFunctionAddress;
  MS_ASSERT(func != nullptr);
  return func(func_name);
}
// clCreateProgramWithIL wrapper, use OpenCLWrapper function.
cl_program clCreateProgramWithIL(cl_context context, const void *il, size_t length, cl_int *ret) {
  auto func = mindspore::lite::opencl::clCreateProgramWithIL;
  MS_ASSERT(func != nullptr);
  return func(context, il, length, ret);
}

// clSVMAlloc wrapper, use OpenCLWrapper function.
void *clSVMAlloc(cl_context context, cl_mem_flags flags, size_t size, cl_uint align) {
  auto func = mindspore::lite::opencl::clSVMAlloc;
  MS_ASSERT(func != nullptr);
  return func(context, flags, size, align);
}

// clSVMFree wrapper, use OpenCLWrapper function.
void clSVMFree(cl_context context, void *buffer) {
  auto func = mindspore::lite::opencl::clSVMFree;
  MS_ASSERT(func != nullptr);
  func(context, buffer);
}

// clEnqueueSVMMap wrapper, use OpenCLWrapper function.
cl_int clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking, cl_map_flags flags, void *host_ptr,
                       size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueSVMMap;
  MS_ASSERT(func != nullptr);
  return func(command_queue, blocking, flags, host_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueSVMUnmap wrapper, use OpenCLWrapper function.
cl_int clEnqueueSVMUnmap(cl_command_queue command_queue, void *host_ptr, cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event) {
  auto func = mindspore::lite::opencl::clEnqueueSVMUnmap;
  MS_ASSERT(func != nullptr);
  return func(command_queue, host_ptr, num_events_in_wait_list, event_wait_list, event);
}

// clSetKernelArgSVMPointer wrapper, use OpenCLWrapper function.
cl_int clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint index, const void *host_ptr) {
  auto func = mindspore::lite::opencl::clSetKernelArgSVMPointer;
  MS_ASSERT(func != nullptr);
  return func(kernel, index, host_ptr);
}
#endif

#endif  // USE_OPENCL_WRAPPER
