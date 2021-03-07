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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_SELF_PARAMETER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_SELF_PARAMETER_H_

#include <vector>
#include <string>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/arithmetic_self_parameter.h"

using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_ExpFusion;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {

class ArithmeticSelfOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~ArithmeticSelfOpenCLKernel() override = default;

  int Prepare() override;

  int CheckSpecs() override;
  void SetConstArgs() override { ocl_runtime_->SetKernelArg(kernel_, 2, output_shape_); }
  void SetGlobalLocal() override;

  int Run() override;

 private:
  cl_int4 output_shape_ = {};
};

}  // namespace mindspore::kernel
#endif
