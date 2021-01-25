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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/anf.h"

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using CTask = std::function<void(size_t, size_t)>;
namespace mindspore {
namespace kernel {
const char KERNEL_SIZE[] = "kernel_size";
const char STRIDE[] = "stride";
const char STRIDES[] = "strides";
const char DILATION[] = "dilation";
const char PAD[] = "pad";
const char PAD_LIST[] = "pad_list";
const char PAD_MODE[] = "pad_mode";
const char PAD_MODE_LOWER_SAME[] = "same";
const char PAD_MODE_LOWER_VALID[] = "valid";
const char PAD_MODE_UPPER_SAME[] = "SAME";
const char PAD_MODE_UPPER_VALID[] = "VALID";
const char TRANSPOSE_A[] = "transpose_a";
const char TRANSPOSE_B[] = "transpose_b";
const char IS_GRAD[] = "is_grad";
const char TRANSPOSE_NO = 'N';
const char TRANSPOSE_YES = 'T';
const char AXIS[] = "axis";
const char DIM[] = "dim";
const char BEGIN[] = "begin";
const char END[] = "end";
const char SIZE[] = "size";
const char USE_NESTEROV[] = "use_nesterov";
const char GROUP[] = "group";
const char START[] = "start";
const char LIMIT[] = "limit";
const char DELTA[] = "delta";

enum OperateType {
  ADD = 0,
  SUB,
  MUL,
  DIV,
  SQUARE,
  SQRT,
  POW,
  REALDIV,
  FLOORDIV,
  MOD,
  NEG,
  LESS,
  ASSIGNADD,
  RELUGRAD,
  RELU6GRAD,
  ABSGRAD,
  TANHGRAD,
  SQRTGRAD,
  SIGMOIDGRAD,
  ONESLIKE,
  ZEROSLIKE,
  SIGN,
  EQUAL,
  NOTEQUAL,
  LESSEQUAL,
  LOGICALAND,
  LOGICALOR,
  LOGICALNOT,
  FLOOR,
  SQUAREDDIFFERENCE,
  GREATER,
  GREATEREQUAL,
  RECIPROCAL,
  GELU,
  GELUGRAD,
};

class CPUKernel : public kernel::KernelMod {
 public:
  CPUKernel() = default;
  ~CPUKernel() override = default;
  virtual void Init(const CNodePtr &kernel_node);
  virtual void InitKernel(const CNodePtr &kernel_node) = 0;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void * /*stream_ptr*/) override {
    return Launch(inputs, workspace, outputs);
  };
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs) = 0;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

 protected:
  virtual void InitInputOutputSize(const CNodePtr &kernel_node);
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

class CPUKernelUtils {
 public:
  static void ExpandDimsTo4(std::vector<size_t> *shape);
  static size_t CalcOffset(const std::vector<size_t> &shape, size_t dim0, size_t dim1, size_t dim2, size_t dim3);
  static size_t GetElementNumOnAxis(const std::vector<size_t> &shape, int axis);
  static void GetElementNumEveryDim(const std::vector<size_t> &shape, std::vector<size_t> *element_num);
  static void ParallelFor(const CTask &task, size_t count);
  static std::vector<size_t> FlatShapeByAxis(const std::vector<size_t> &shape, int axis);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CPU_KERNEL_H_
