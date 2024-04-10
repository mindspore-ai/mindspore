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
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <string>
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"
#include "src/common/log.h"
#include "include/errorcode.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"
#include "src/common/common.h"
namespace mindspore::ascend_native {

int vec_core_num;
int cube_core_num;

void PrintFp16Host(void *x, size_t elem_num, void *stream) {
  float *dst_h = reinterpret_cast<float *>(malloc(sizeof(float) * elem_num));
  void *dst_d = MallocDevice(sizeof(float) * elem_num);
  ascend_native::CopyHostFp16ToDeviceFp32(x, &dst_d, elem_num, stream);
  ascend_native::CopyDeviceFp32ToHostFp32(dst_d, reinterpret_cast<void *>(dst_h), elem_num, stream);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << dst_h[i] << " ";
  }
  std::cout << std::endl;
  free(dst_h);
  ascend_native::FreeDevice(dst_d);
}

void PrintFp16(void *x, size_t elem_num, void *stream) {
  float *ptr = reinterpret_cast<float *>(malloc(sizeof(float) * elem_num));
  ascend_native::CopyDeviceFp16ToHostFp32(x, reinterpret_cast<void *>(ptr), elem_num, stream);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
  free(ptr);
}

void PrintFp32(void *x, size_t elem_num, void *stream) {
  std::vector<float> ptr(elem_num);
  ascend_native::CopyDeviceFp32ToHostFp32(x, ptr.data(), elem_num, stream);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

void PrintInt32(void *x, size_t elem_num, void *stream) {
  std::vector<int> ptr(elem_num);
  ascend_native::CopyDeviceFp32ToHostFp32(x, ptr.data(), elem_num, stream);
  for (size_t i = 0; i < elem_num; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void printVectorChecksum(void *x, int elem_num, void *q) {
  auto ptr = std::vector<float>(elem_num);
  float sum = 0;
  ascend_native::CopyDeviceFp16ToHostFp32(x, ptr.data(), elem_num, q);
  ascend_native::SyncDevice(q);
  for (auto &item : ptr) {
    sum += item;
  }
  std::cout << "checksum: " << static_cast<float>(sum) << std::endl;
}

bool IsNaN(float *val) {
  uint32_t temp;
  std::memcpy(&temp, val, sizeof(float));
  return (temp & 0x7FFFFFFF) > 0x7F800000;
}

void vectorCheckNanOrInf(void *x, int elem_num, char *tensor_name, void *stream) {
  float *ptr = reinterpret_cast<float *>(malloc(sizeof(float) * elem_num));
  ascend_native::CopyDeviceFp16ToHostFp32(x, reinterpret_cast<void *>(ptr), elem_num, stream);
  bool is_nan_or_inf = false;
  for (int i = 0; i < elem_num; i++) {
    if (IsNaN(&ptr[i])) {
      std::cout << "value in index: " << i << " in tensor: " << tensor_name << " is nan" << std::endl;
      is_nan_or_inf = true;
    }
    if (std::isinf(ptr[i])) {
      std::cout << "value in index: " << i << " in tensor: " << tensor_name << " is inf" << std::endl;
      is_nan_or_inf = true;
    }
  }
  if (!is_nan_or_inf)
    std::cout << "tensor: " << tensor_name << ", size = " << elem_num << " does not contain nan or inf value"
              << std::endl;
  free(ptr);
}
void *MallocDevice(size_t size) {
  void *device_data = nullptr;
  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&device_data), ALIGN32(size), ACL_MEM_MALLOC_HUGE_FIRST));
  return device_data;
}

void FreeDevice(void *ptr) {
  if (ptr != nullptr) {
    CHECK_ACL(aclrtFree(ptr));
  }
}

void SyncDevice(void *stream) { CHECK_ACL(aclrtSynchronizeStream(stream)); }

int InitAcl() {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    return ret;
  }
  initCoresNum();
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

void DestroyStream(void *stream) { CHECK_ACL(aclrtDestroyStream(stream)); }

void SetContext(void *ctx) {
  if (ctx != nullptr) CHECK_ACL(aclrtSetCurrentContext(ctx));
}

void *CreateStream(void *context) {
  aclrtStream stream = nullptr;
  aclrtSetCurrentContext(context);
  CHECK_ACL(aclrtCreateStream(&stream));
  return stream;
}

void *CreateContext(int32_t deviceId) {
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtContext context;
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  return context;
}

// host to device
void *MallocCopy(void *src, size_t size) {
  void *ptr = MallocDevice(size);
  ascend_native::CopyHTD(ptr, src, size);
  return ptr;
}
// copy host to device
void CopyHTD(void *dst, void *src, size_t size) {
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

// copy device to host
void CopyDTH(void *dst, void *src, size_t size) {
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
}
int getVecNum() { return vec_core_num; }

int getCubeNum() { return cube_core_num; }
int initCoresNum() {
  const char *soc_name_ptr = aclrtGetSocName();
  if (soc_name_ptr == nullptr) {
    MS_LOG(ERROR) << "aclrtGetSocName failed";
    return ACL_ERROR_NONE;
  }
  if (std::string(soc_name_ptr) == std::string("Ascend910B1")) {
    vec_core_num = VEC_CORE_NUM_910B1;
    cube_core_num = CUBE_CORE_NUM_910B1;
  } else if (std::string(soc_name_ptr) == std::string("Ascend910B4")) {
    vec_core_num = VEC_CORE_NUM_910B4;
    cube_core_num = CUBE_CORE_NUM_910B4;
  } else {
    MS_LOG(ERROR) << "soc_name isn't Ascend910B1 or Ascend910B4";
  }
  return 0;
}

uint64_t GetTimeUs() {
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  auto ret_val = static_cast<uint64_t>((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return ret_val;
}

int MemGetInfo(size_t *free, size_t *total) {
  aclError ret = aclrtGetMemInfo(ACL_HBM_MEM_HUGE, free, total);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclError MemGetInfo" << ret;
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

std::vector<int64_t> calcStride(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - Num2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

void saveTensor(const std::string &name, int cnt, void *tensor, int size) {
  void *ptr = malloc(size);
  char c = '0' + cnt;
  CopyDTH(ptr, tensor, size);
  std::ofstream wf(name + c + ".bin", std::ofstream::out | std::ofstream::binary);
  wf.write(reinterpret_cast<char *>(ptr), size);
  wf.close();
}
}  // namespace mindspore::ascend_native
