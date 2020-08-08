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

#ifndef MINDSPORE_LITE_SRC_LITE_KERNEL_H_
#define MINDSPORE_LITE_SRC_LITE_KERNEL_H_
#include <vector>
#include <string>
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "include/context.h"
#include "src/ir/tensor.h"
#include "src/ops/ops.h"

#ifdef ENABLE_FP16
using FLOAT_t = float16_t;
#else
using FLOAT_t = float;
#endif

// using mindspore::kernel::AddressPtr;
namespace mindspore::kernel {
enum KERNEL_ARCH { kCPU, kGPU, kNPU, kKernelArch_MIN = kCPU, kKernelArch_MAX = kNPU };
struct KernelKey {
  KERNEL_ARCH arch;
  TypeId data_type;
  schema::PrimitiveType type;

  bool operator<(const KernelKey &dst) const {
    if (arch != dst.arch) {
      return arch < dst.arch;
    } else if (data_type != dst.data_type) {
      return data_type < dst.data_type;
    } else {
      return type < dst.type;
    }
  }
};

class LiteKernel {
 public:
  LiteKernel() = default;
  explicit LiteKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs)
      : opParameter(parameter), inputs_(inputs), outputs_(outputs), train_mode(false) {
    this->in_kernel_.clear();
    this->out_kernel_.clear();
  }

  virtual ~LiteKernel() { delete opParameter; }

  virtual int Prepare() { return -1; }
  virtual int Init() { return -1; }
  virtual int ReSize() { return -1; }
  virtual int Run() { return -1; }

  std::string Name() { return this->name; }
  virtual void train() { train_mode = true; }
  virtual bool is_train() { return train_mode == true; }
  virtual void eval() { train_mode = false; }
  virtual bool is_eval() { return train_mode == false; }
  void set_name(const std::string &name) { this->name = name; }

  schema::PrimitiveType type() { return (schema::PrimitiveType)this->opParameter->type_; }

  std::string type_str() { return schema::EnumNamePrimitiveType((schema::PrimitiveType)this->opParameter->type_); }

  void SetInputs(const std::vector<lite::tensor::Tensor *> &inputs) { this->inputs_ = inputs; }

  void SetOutputs(const std::vector<lite::tensor::Tensor *> &outputs) { this->outputs_ = outputs; }

  std::vector<lite::tensor::Tensor *> &GetInputs() { return this->inputs_; }

  std::vector<lite::tensor::Tensor *> &GetOutputs() { return this->outputs_; }

  void AddInKernel(LiteKernel *kernel) { this->in_kernel_.emplace_back(kernel); }

  void AddOutKernel(LiteKernel *kernel) { this->out_kernel_.emplace_back(kernel); }

  std::vector<LiteKernel *> &GetInKernels() { return this->in_kernel_; }

  std::vector<LiteKernel *> &GetOutKernels() { return this->out_kernel_; }

  void InitOutTensorRefCount();

  int DecOutTensorRefCount();

  const KernelKey Desc() const { return desc; }

  void set_desc(const KernelKey kernel_key) { desc = kernel_key; }

 protected:
  KernelKey desc;
  std::string name;
  OpParameter *opParameter = nullptr;
  // tensor will free in ~lite_session()
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  std::vector<LiteKernel *> in_kernel_;
  std::vector<LiteKernel *> out_kernel_;
  bool train_mode;
};

class SubGraphKernel : public LiteKernel {
 public:
  explicit SubGraphKernel(const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs,
                          const std::vector<kernel::LiteKernel *> &inKernels,
                          const std::vector<kernel::LiteKernel *> &outKernels,
                          const std::vector<kernel::LiteKernel *> &nodes)
      : LiteKernel(nullptr, inputs, outputs),
        inputs_(inputs),
        outputs_(outputs),
        inkernels_(inKernels),
        outkernels_(outKernels),
        nodes_(nodes) {}

  virtual int Init() { return -1; }
  virtual int InferShape() { return -1; }
  virtual int ReSize() { return -1; }
  virtual int Run() { return -1; }

 protected:
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  std::vector<LiteKernel *> inkernels_;
  std::vector<LiteKernel *> outkernels_;
  std::vector<LiteKernel *> nodes_;
};

typedef LiteKernel *(*KernelCreator)(const std::vector<lite::tensor::Tensor *> &inputs,
                                     const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                     const lite::Context *ctx, const KernelKey &desc);

class LiteKernelUtil {
 public:
  static void TopologicalSortKernels(std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<kernel::LiteKernel *> SubgraphInputKernels(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<kernel::LiteKernel *> SubgraphOutputKernels(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<lite::tensor::Tensor *> SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<lite::tensor::Tensor *> SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels);

  static void InitTensorRefCount(std::vector<kernel::LiteKernel *> &kernels);

  static int SetInput(LiteKernel &kernelMod, std::vector<lite::tensor::Tensor *> inputs);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITE_KERNEL_H_
