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
#include "include/errorcode.h"

#ifdef ENABLE_FP16
using FLOAT_t = float16_t;
#else
using FLOAT_t = float;
#endif

// using mindspore::kernel::AddressPtr;
namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
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
  explicit LiteKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &in_tensors,
                      const std::vector<lite::tensor::Tensor *> &out_tensors, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : op_parameter_(parameter),
        in_tensors_(in_tensors),
        out_tensors_(out_tensors),
        primitive_(primitive),
        context_(ctx) {
    if (op_parameter_ && ctx) {
      op_parameter_->thread_num_ = ctx->thread_num_;
    }
    this->in_kernels_.clear();
    this->out_kernels_.clear();
  }

  virtual ~LiteKernel() { delete op_parameter_; }

  virtual int Prepare() {
    if (!InferShapeDone()) {
      (const_cast<lite::Primitive *>(primitive_))->InferShape(in_tensors_, out_tensors_);
      if (need_reinit_) {
        Init();
      }
    }

    auto &outputs = this->out_tensors();
    for (auto *output : outputs) {
      MS_ASSERT(output != nullptr);
      output->MallocData();
    }
    return RET_OK;
  }

  virtual int Init() { return -1; }

  virtual int ReSize() { return -1; }

  virtual int Run() { return -1; }

  std::string name() { return this->name_; }

  virtual void train() { train_mode_ = true; }

  virtual bool is_train() { return train_mode_; }

  virtual void eval() { train_mode_ = false; }

  virtual bool is_eval() { return !train_mode_; }

  void set_name(const std::string &name) { this->name_ = name; }

  void set_is_model_output(bool is_model_output) { this->is_model_output_ = is_model_output; }

  bool is_model_output() const { return this->is_model_output_; }

  schema::PrimitiveType Type() {
    return (this->op_parameter_ != nullptr) ? schema::PrimitiveType(this->op_parameter_->type_)
                                            : schema::PrimitiveType_NONE;
  }

  std::string type_str() { return schema::EnumNamePrimitiveType(this->Type()); }

  void set_in_tensors(const std::vector<lite::tensor::Tensor *> &in_tensors) { this->in_tensors_ = in_tensors; }

  void set_out_tensors(const std::vector<lite::tensor::Tensor *> &out_tensors) { this->out_tensors_ = out_tensors; }

  std::vector<lite::tensor::Tensor *> &in_tensors() { return this->in_tensors_; }

  std::vector<lite::tensor::Tensor *> &out_tensors() { return this->out_tensors_; }

  void AddInKernel(LiteKernel *kernel) { this->in_kernels_.emplace_back(kernel); }

  void AddOutKernel(LiteKernel *kernel) { this->out_kernels_.emplace_back(kernel); }

  std::vector<LiteKernel *> &in_kernels() { return this->in_kernels_; }

  std::vector<LiteKernel *> &out_kernels() { return this->out_kernels_; }

  void InitOutTensorRefCount();

  int DecOutTensorRefCount();

  KernelKey desc() const { return desc_; }

  void set_desc(const KernelKey kernel_key) { desc_ = kernel_key; }

  void set_need_reinit() { need_reinit_ = true; }

 protected:
  bool InferShapeDone() { return !(primitive_ != nullptr && !primitive_->GetInferFlag()) && true; }

  KernelKey desc_;
  std::string name_;
  OpParameter *op_parameter_ = nullptr;
  const lite::Primitive *primitive_ = nullptr;
  const lite::Context *context_ = nullptr;
  // tensor will free in ~lite_session()
  std::vector<lite::tensor::Tensor *> in_tensors_;
  std::vector<lite::tensor::Tensor *> out_tensors_;
  std::vector<LiteKernel *> in_kernels_;
  std::vector<LiteKernel *> out_kernels_;
  bool train_mode_ = false;
  bool need_reinit_ = false;
  bool is_model_output_ = false;
};

class SubGraphKernel : public LiteKernel {
 public:
  explicit SubGraphKernel(const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs,
                          const std::vector<kernel::LiteKernel *> &inKernels,
                          const std::vector<kernel::LiteKernel *> &outKernels,
                          const std::vector<kernel::LiteKernel *> &nodes, const lite::Context *ctx,
                          const lite::Primitive *primitive)
      : LiteKernel(nullptr, inputs, outputs, ctx, primitive),
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
                                     const lite::Context *ctx, const KernelKey &desc, const lite::Primitive *primitive);

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
