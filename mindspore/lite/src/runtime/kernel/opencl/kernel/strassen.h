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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STRASSEN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STRASSEN_H_

#include <string>
#include <vector>
#include "src/runtime/kernel/opencl/kernel/matmul.h"

#define MAXDEPTH 5

namespace mindspore::kernel {
class StrassenOpenCLKernel : public MatMulOpenCLKernel {
 public:
  using MatMulOpenCLKernel::MatMulOpenCLKernel;
  ~StrassenOpenCLKernel() override = default;

 public:
  int Run() override;
  int Prepare() override;
  int InitWeights() override;
  int SetConstArgs() override;
  void SetGlobalLocal() override;

  // strassen
 private:
  int AllocatorMemoryForStrassen(int NumA, int NumB);
  int DoStrassen(void *data, void *weight, void *result, const int size, const int depth, const int threshold);
  int StrassenSetGlobalLocal(size_t strassen_size, int type_flag);
  int StrassenSetConstArgs(cl::Kernel *kernel, int index, int strassen_size, bool is_matmul_kernel);
  int StrassenDataFilled(cl::Kernel *kernel, void *input, void *output, const int size, cl_int2 offset,
                         lite::opencl::MemType mem_type);
  int StrassenAddSub(cl::Kernel *kernel, void *input, void *output, const int size, cl_int4 offset, int flag,
                     lite::opencl::MemType mem_type);
  int StrassenBackResult(cl::Kernel *kernel, void *input1, void *input2, void *input3, void *input4, void *input5,
                         void *input6, void *input7, void *output, const int size);
  int StrassenRunMmatmul(void *input, void *weight, void *output, const int size);
  cl::Kernel kernel_IMG_add_sub_2;
  cl::Kernel MatMul_StrassenBUFFilled;
  cl::Kernel MatMul_StrassenIMGFilled;
  cl::Kernel kernel_BUF_add_sub_2;
  cl::Kernel kernel_back_result;
  cl::NDRange global_add_sub_, local_add_sub_;
  std::vector<size_t> global_size_add_sub;
  std::vector<size_t> local_size_add_sub;
  // image 2d
  void *A_temp[MAXDEPTH] = {nullptr};
  void *M1[MAXDEPTH] = {nullptr};
  void *M2[MAXDEPTH] = {nullptr};
  void *M3[MAXDEPTH] = {nullptr};
  void *M4[MAXDEPTH] = {nullptr};
  void *M5[MAXDEPTH] = {nullptr};
  void *M6[MAXDEPTH] = {nullptr};
  void *M7[MAXDEPTH] = {nullptr};
  // buffer
  void *B_temp[MAXDEPTH] = {nullptr};
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_WINOGRAD_H_
