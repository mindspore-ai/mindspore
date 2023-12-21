/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"

namespace mindspore::ascend_native {

void PrintFp16Host(void *x, size_t elem_num, void *stream, void *ctx) {
  float *dst_h = reinterpret_cast<float *>(malloc(sizeof(float) * elem_num));
  void *dst_d = MallocDevice(sizeof(float) * elem_num, ctx);
  ascend_native::CopyHostFp16ToDeviceFp32(x, &dst_d, elem_num, stream, ctx);
  ascend_native::CopyDeviceFp32ToHostFp32(dst_d, reinterpret_cast<void *>(dst_h), elem_num, stream, ctx);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << dst_h[i] << " ";
  }
  std::cout << std::endl;
  free(dst_h);
  ascend_native::FreeDevice(dst_d, ctx);
}

void PrintFp16(void *x, size_t elem_num, void *stream, void *ctx) {
  float *ptr = reinterpret_cast<float *>(malloc(sizeof(float) * elem_num));
  ascend_native::CopyDeviceFp16ToHostFp32(x, reinterpret_cast<void *>(ptr), elem_num, stream, ctx);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
  free(ptr);
}

void PrintFp32(void *x, size_t elem_num, void *stream, void *ctx) {
  std::vector<float> ptr(elem_num);
  ascend_native::CopyDeviceFp32ToHostFp32(x, ptr.data(), elem_num, stream, ctx);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

void PrintInt32(void *x, size_t elem_num, void *stream, void *ctx) {
  std::vector<int> ptr(elem_num);
  ascend_native::CopyDeviceFp32ToHostFp32(x, ptr.data(), elem_num, stream, ctx);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void printVectorChecksum(void *x, int elem_num, void *q, void *ctx) {
  auto ptr = std::vector<float>(elem_num);
  float sum = 0;
  ascend_native::CopyDeviceFp16ToHostFp32(x, ptr.data(), elem_num, q, ctx);
  ascend_native::SyncDevice(q, ctx);
  for (auto &item : ptr) {
    sum += item;
  }
  std::cout << "checksum: " << static_cast<float>(sum) << std::endl;
}

void *MallocDevice(size_t size, void *ctx) {
  SetContext(ctx);
  void *device_data = nullptr;
  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&device_data), ALIGN32(size), ACL_MEM_MALLOC_HUGE_FIRST));
  return device_data;
}

void FreeDevice(void *ptr, void *ctx) {
  if (ptr != nullptr) {
    CHECK_ACL(aclrtFree(ptr));
  }
}

void SyncDevice(void *stream, void *ctx) {
  SetContext(ctx);
  CHECK_ACL(aclrtSynchronizeStream(stream));
}

int InitAcl() {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    return ret;
  }
  return ACL_SUCCESS;
}

int FinalizeAcl() {
  aclError ret = aclFinalize();
  if (ret != ACL_ERROR_NONE) {
    return ret;
  }
  return ACL_SUCCESS;
}

void DestroyCtx(void *context) { CHECK_ACL(aclrtDestroyContext(context)); }

void DestroyStream(void *stream, void *ctx) {
  SetContext(ctx);
  CHECK_ACL(aclrtDestroyStream(stream));
}

void *CreateStream(void *context) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));
  return stream;
}

void *CreateContext(int32_t deviceId) {
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtContext context;
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  return context;
}

void SetContext(void *ctx) { CHECK_ACL(aclrtSetCurrentContext(ctx)); }

// host to device
void *MallocCopy(void *src, size_t size, void *ctx) {
  void *ptr = MallocDevice(size, ctx);
  SetContext(ctx);
  ascend_native::CopyHTD(ptr, src, size, ctx);
  return ptr;
}
// copy host to device
void CopyHTD(void *dst, void *src, size_t size, void *ctx) {
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

// copy device to host
void CopyDTH(void *dst, void *src, size_t size, void *ctx) {
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

uint64_t GetTimeUs() {
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  auto ret_val = static_cast<uint64_t>((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return ret_val;
}
}  // namespace mindspore::ascend_native
