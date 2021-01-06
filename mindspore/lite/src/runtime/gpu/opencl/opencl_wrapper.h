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

#ifndef MINDSPORE_LITE_SRC_OPENCL_WRAPPER_H_
#define MINDSPORE_LITE_SRC_OPENCL_WRAPPER_H_

#include <memory>
#include <string>
#include <algorithm>
#include "CL/cl2.hpp"

#ifdef USE_OPENCL_WRAPPER

namespace mindspore::lite::opencl {
// This is a opencl function wrapper.
bool LoadOpenCLLibrary(void **handle_ptr);
bool UnLoadOpenCLLibrary(void *handle);

// get platform id
using clGetPlatformIDsFunc = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
// get platform info
using clGetPlatformInfoFunc = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
// build program
using clBuildProgramFunc = cl_int (*)(cl_program, cl_uint, const cl_device_id *, const char *,
                                      void (*pfn_notify)(cl_program, void *), void *);
// enqueue run kernel
using clEnqueueNDRangeKernelFunc = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                              const size_t *, cl_uint, const cl_event *, cl_event *);
// set kernel parameter
using clSetKernelArgFunc = cl_int (*)(cl_kernel, cl_uint, size_t, const void *);
using clRetainMemObjectFunc = cl_int (*)(cl_mem);
using clReleaseMemObjectFunc = cl_int (*)(cl_mem);
using clEnqueueUnmapMemObjectFunc = cl_int (*)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
using clRetainCommandQueueFunc = cl_int (*)(cl_command_queue command_queue);
// create context
using clCreateContextFunc = cl_context (*)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                           void(CL_CALLBACK *)(  // NOLINT(readability/casting)
                                             const char *, const void *, size_t, void *),
                                           void *, cl_int *);
using clEnqueueCopyImageFunc = cl_int (*)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *,
                                          const size_t *, cl_uint, const cl_event *, cl_event *);

using clCreateContextFromTypeFunc = cl_context (*)(const cl_context_properties *, cl_device_type,
                                                   void(CL_CALLBACK *)(  // NOLINT(readability/casting)
                                                     const char *, const void *, size_t, void *),
                                                   void *, cl_int *);
using clReleaseContextFunc = cl_int (*)(cl_context);
using clWaitForEventsFunc = cl_int (*)(cl_uint, const cl_event *);
using clReleaseEventFunc = cl_int (*)(cl_event);
using clEnqueueWriteBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint,
                                            const cl_event *, cl_event *);
using clEnqueueWriteImageFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t,
                                           size_t, const void *, cl_uint, const cl_event *, cl_event *);
using clEnqueueReadImageFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t,
                                          size_t, void *, cl_uint, const cl_event *, cl_event *);
using clEnqueueReadBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint,
                                           const cl_event *, cl_event *);
using clGetProgramBuildInfoFunc = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
using clRetainProgramFunc = cl_int (*)(cl_program program);
using clEnqueueMapBufferFunc = void *(*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint,
                                         const cl_event *, cl_event *, cl_int *);
using clEnqueueMapImageFunc = void *(*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *,
                                        size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);
using clCreateCommandQueueFunc = cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
using clGetCommandQueueInfoFunc = cl_int (*)(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
using clReleaseCommandQueueFunc = cl_int (*)(cl_command_queue);
using clCreateProgramWithBinaryFunc = cl_program (*)(cl_context, cl_uint, const cl_device_id *, const size_t *,
                                                     const unsigned char **, cl_int *, cl_int *);
using clRetainContextFunc = cl_int (*)(cl_context context);
using clGetContextInfoFunc = cl_int (*)(cl_context, cl_context_info, size_t, void *, size_t *);
using clReleaseProgramFunc = cl_int (*)(cl_program program);
using clFlushFunc = cl_int (*)(cl_command_queue command_queue);
using clFinishFunc = cl_int (*)(cl_command_queue command_queue);
using clGetProgramInfoFunc = cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
using clCreateKernelFunc = cl_kernel (*)(cl_program, const char *, cl_int *);
using clRetainKernelFunc = cl_int (*)(cl_kernel kernel);
using clCreateBufferFunc = cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
using clCreateImage2DFunc = cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t,
                                       void *, cl_int *);
using clCreateImage3DFunc = cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t,
                                       size_t, size_t, void *, cl_int *);
using clCreateProgramWithSourceFunc = cl_program (*)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
using clReleaseKernelFunc = cl_int (*)(cl_kernel kernel);
using clGetDeviceInfoFunc = cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
using clGetDeviceIDsFunc = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
using clRetainEventFunc = cl_int (*)(cl_event);
using clGetKernelWorkGroupInfoFunc = cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *,
                                                size_t *);
using clGetEventInfoFunc = cl_int (*)(cl_event event, cl_event_info param_name, size_t param_value_size,
                                      void *param_value, size_t *param_value_size_ret);
using clGetEventProfilingInfoFunc = cl_int (*)(cl_event event, cl_profiling_info param_name, size_t param_value_size,
                                               void *param_value, size_t *param_value_size_ret);
using clGetImageInfoFunc = cl_int (*)(cl_mem, cl_image_info, size_t, void *, size_t *);
using clEnqueueCopyBufferToImageFunc = cl_int (*)(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *,
                                                  const size_t *, cl_uint, const cl_event *, cl_event *);
using clEnqueueCopyImageToBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *,
                                                  size_t, cl_uint, const cl_event *, cl_event *);
#if CL_TARGET_OPENCL_VERSION >= 120
using clRetainDeviceFunc = cl_int (*)(cl_device_id);
using clReleaseDeviceFunc = cl_int (*)(cl_device_id);
using clCreateImageFunc = cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *,
                                     cl_int *);
using clEnqueueFillImageFunc = cl_int (*)(cl_command_queue, cl_mem, const void *, const size_t *, const size_t *,
                                          cl_uint, const cl_event *, cl_event *);
#endif
#if CL_TARGET_OPENCL_VERSION >= 200
using clCreateProgramWithILFunc = cl_program (*)(cl_context, const void *, size_t, cl_int *);
using clSVMAllocFunc = void *(*)(cl_context, cl_mem_flags, size_t size, cl_uint);
using clSVMFreeFunc = void (*)(cl_context, void *);
using clEnqueueSVMMapFunc = cl_int (*)(cl_command_queue, cl_bool, cl_map_flags, void *, size_t, cl_uint,
                                       const cl_event *, cl_event *);
using clEnqueueSVMUnmapFunc = cl_int (*)(cl_command_queue, void *, cl_uint, const cl_event *, cl_event *);
using clSetKernelArgSVMPointerFunc = cl_int (*)(cl_kernel, cl_uint, const void *);
// opencl 2.0 can get sub group info and wave size.
using clGetKernelSubGroupInfoKHRFunc = cl_int (*)(cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t,
                                                  const void *, size_t, void *, size_t *);
using clCreateCommandQueueWithPropertiesFunc = cl_command_queue (*)(cl_context, cl_device_id,
                                                                    const cl_queue_properties *, cl_int *);
using clGetExtensionFunctionAddressFunc = void *(*)(const char *);
#endif

#define CL_DECLARE_FUNC_PTR(func) extern func##Func func

CL_DECLARE_FUNC_PTR(clGetPlatformIDs);
CL_DECLARE_FUNC_PTR(clGetPlatformInfo);
CL_DECLARE_FUNC_PTR(clBuildProgram);
CL_DECLARE_FUNC_PTR(clEnqueueNDRangeKernel);
CL_DECLARE_FUNC_PTR(clSetKernelArg);
CL_DECLARE_FUNC_PTR(clReleaseKernel);
CL_DECLARE_FUNC_PTR(clCreateProgramWithSource);
CL_DECLARE_FUNC_PTR(clCreateBuffer);
CL_DECLARE_FUNC_PTR(clCreateImage2D);
CL_DECLARE_FUNC_PTR(clCreateImage3D);
CL_DECLARE_FUNC_PTR(clRetainKernel);
CL_DECLARE_FUNC_PTR(clCreateKernel);
CL_DECLARE_FUNC_PTR(clGetProgramInfo);
CL_DECLARE_FUNC_PTR(clFlush);
CL_DECLARE_FUNC_PTR(clFinish);
CL_DECLARE_FUNC_PTR(clReleaseProgram);
CL_DECLARE_FUNC_PTR(clRetainContext);
CL_DECLARE_FUNC_PTR(clGetContextInfo);
CL_DECLARE_FUNC_PTR(clCreateProgramWithBinary);
CL_DECLARE_FUNC_PTR(clCreateCommandQueue);
CL_DECLARE_FUNC_PTR(clGetCommandQueueInfo);
CL_DECLARE_FUNC_PTR(clReleaseCommandQueue);
CL_DECLARE_FUNC_PTR(clEnqueueMapBuffer);
CL_DECLARE_FUNC_PTR(clEnqueueMapImage);
CL_DECLARE_FUNC_PTR(clEnqueueCopyImage);
CL_DECLARE_FUNC_PTR(clRetainProgram);
CL_DECLARE_FUNC_PTR(clGetProgramBuildInfo);
CL_DECLARE_FUNC_PTR(clEnqueueReadBuffer);
CL_DECLARE_FUNC_PTR(clEnqueueWriteBuffer);
CL_DECLARE_FUNC_PTR(clEnqueueWriteImage);
CL_DECLARE_FUNC_PTR(clEnqueueReadImage);
CL_DECLARE_FUNC_PTR(clWaitForEvents);
CL_DECLARE_FUNC_PTR(clReleaseEvent);
CL_DECLARE_FUNC_PTR(clCreateContext);
CL_DECLARE_FUNC_PTR(clCreateContextFromType);
CL_DECLARE_FUNC_PTR(clReleaseContext);
CL_DECLARE_FUNC_PTR(clRetainCommandQueue);
CL_DECLARE_FUNC_PTR(clEnqueueUnmapMemObject);
CL_DECLARE_FUNC_PTR(clRetainMemObject);
CL_DECLARE_FUNC_PTR(clReleaseMemObject);
CL_DECLARE_FUNC_PTR(clGetDeviceInfo);
CL_DECLARE_FUNC_PTR(clGetDeviceIDs);
CL_DECLARE_FUNC_PTR(clRetainEvent);
CL_DECLARE_FUNC_PTR(clGetKernelWorkGroupInfo);
CL_DECLARE_FUNC_PTR(clGetEventInfo);
CL_DECLARE_FUNC_PTR(clGetEventProfilingInfo);
CL_DECLARE_FUNC_PTR(clGetImageInfo);
CL_DECLARE_FUNC_PTR(clEnqueueCopyBufferToImage);
CL_DECLARE_FUNC_PTR(clEnqueueCopyImageToBuffer);
#if CL_TARGET_OPENCL_VERSION >= 120
CL_DECLARE_FUNC_PTR(clRetainDevice);
CL_DECLARE_FUNC_PTR(clReleaseDevice);
CL_DECLARE_FUNC_PTR(clCreateImage);
CL_DECLARE_FUNC_PTR(clEnqueueFillImage);
#endif
#if CL_TARGET_OPENCL_VERSION >= 200
CL_DECLARE_FUNC_PTR(clGetKernelSubGroupInfoKHR);
CL_DECLARE_FUNC_PTR(clCreateCommandQueueWithProperties);
CL_DECLARE_FUNC_PTR(clGetExtensionFunctionAddress);
CL_DECLARE_FUNC_PTR(clCreateProgramWithIL);
CL_DECLARE_FUNC_PTR(clSVMAlloc);
CL_DECLARE_FUNC_PTR(clSVMFree);
CL_DECLARE_FUNC_PTR(clEnqueueSVMMap);
CL_DECLARE_FUNC_PTR(clEnqueueSVMUnmap);
CL_DECLARE_FUNC_PTR(clSetKernelArgSVMPointer);
#endif

#undef CL_DECLARE_FUNC_PTR

}  // namespace mindspore::lite::opencl
#endif  // USE_OPENCL_WRAPPER
#endif  // MINDSPORE_LITE_SRC_OPENCL_WRAPPER_H_
