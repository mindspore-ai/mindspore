/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <cuda.h>

CUresult cuModuleLoadData(CUmodule *module, const void *image) { return CUDA_SUCCESS; }

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) { return CUDA_SUCCESS; }

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) { return CUDA_SUCCESS; }

CUresult cuGetErrorName(CUresult error, const char **pStr) { return CUDA_SUCCESS; }

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, int dev) {
  *pi = 0;
  return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) { return CUDA_SUCCESS; }