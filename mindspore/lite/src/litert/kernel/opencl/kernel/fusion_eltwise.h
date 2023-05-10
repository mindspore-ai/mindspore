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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FUSION_ELTWISE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FUSION_ELTWISE_H_

#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include "src/litert/kernel/opencl/opencl_kernel.h"
#include "src/litert/kernel/opencl/kernel/arithmetic.h"
#include "src/litert/kernel/opencl/kernel/arithmetic_self.h"
#include "src/litert/kernel/opencl/kernel/to_format.h"
#include "schema/ops_generated.h"
#include "nnacl/arithmetic_parameter.h"

using mindspore::schema::ActivationType;
using mindspore::schema::PrimitiveType;

namespace mindspore::kernel {
constexpr schema::PrimitiveType PrimitiveType_FusionEltwise = static_cast<schema::PrimitiveType>(-100);

enum EltwiseOperator {
  // Arithmetic Primitive
  Operator_Mul = schema::PrimitiveType_MulFusion,
  Operator_Add = schema::PrimitiveType_AddFusion,
  Operator_Sub = schema::PrimitiveType_SubFusion,
  Operator_Div = schema::PrimitiveType_DivFusion,
  Operator_LogicalAnd = schema::PrimitiveType_LogicalAnd,
  Operator_LogicalOr = schema::PrimitiveType_LogicalOr,
  Operator_Maximum = schema::PrimitiveType_Maximum,
  Operator_Minimum = schema::PrimitiveType_Minimum,
  Operator_FloorDiv = schema::PrimitiveType_FloorDiv,
  Operator_FloorMod = schema::PrimitiveType_FloorMod,
  Operator_SquaredDifference = schema::PrimitiveType_SquaredDifference,
  Operator_Equal = schema::PrimitiveType_Equal,
  Operator_NotEqual = schema::PrimitiveType_NotEqual,
  Operator_Less = schema::PrimitiveType_Less,
  Operator_LessEqual = schema::PrimitiveType_LessEqual,
  Operator_Greater = schema::PrimitiveType_Greater,
  Operator_GreaterEqual = schema::PrimitiveType_GreaterEqual,
  Operator_Eltwise = schema::PrimitiveType_Eltwise,

  // ArithmeticSelf Primitive
  Operator_Abs = schema::PrimitiveType_Abs,
  Operator_Ceil = schema::PrimitiveType_Ceil,
  Operator_Cos = schema::PrimitiveType_Cos,
  Operator_Exp = schema::PrimitiveType_ExpFusion,
  Operator_Floor = schema::PrimitiveType_Floor,
  Operator_Log = schema::PrimitiveType_Log,
  Operator_LogicalNot = schema::PrimitiveType_LogicalNot,
  Operator_Round = schema::PrimitiveType_Round,
  Operator_Rsqrt = schema::PrimitiveType_Rsqrt,
  Operator_Sin = schema::PrimitiveType_Sin,
  Operator_Neg = schema::PrimitiveType_Neg,
  Operator_Sqrt = schema::PrimitiveType_Sqrt,
  Operator_Square = schema::PrimitiveType_Square,

  // Other Primitive
  Operator_Scale = schema::PrimitiveType_ScaleFusion,

  // Activation
  Operator_Act_NO_ACTIVATION = schema::ActivationType_NO_ACTIVATION + schema::PrimitiveType_MAX,
  Operator_Act_RELU = schema::ActivationType_RELU + schema::PrimitiveType_MAX,
  Operator_Act_SIGMOID = schema::ActivationType_SIGMOID + schema::PrimitiveType_MAX,
  Operator_Act_RELU6 = schema::ActivationType_RELU6 + schema::PrimitiveType_MAX,
  Operator_Act_ELU = schema::ActivationType_ELU + schema::PrimitiveType_MAX,
  Operator_Act_LEAKY_RELU = schema::ActivationType_LEAKY_RELU + schema::PrimitiveType_MAX,
  Operator_Act_ABS = schema::ActivationType_ABS + schema::PrimitiveType_MAX,
  Operator_Act_RELU1 = schema::ActivationType_RELU1 + schema::PrimitiveType_MAX,
  Operator_Act_SOFTSIGN = schema::ActivationType_SOFTSIGN + schema::PrimitiveType_MAX,
  Operator_Act_SOFTPLUS = schema::ActivationType_SOFTPLUS + schema::PrimitiveType_MAX,
  Operator_Act_TANH = schema::ActivationType_TANH + schema::PrimitiveType_MAX,
  Operator_Act_SELU = schema::ActivationType_SELU + schema::PrimitiveType_MAX,
  Operator_Act_HSWISH = schema::ActivationType_HSWISH + schema::PrimitiveType_MAX,
  Operator_Act_HSIGMOID = schema::ActivationType_HSIGMOID + schema::PrimitiveType_MAX,
  Operator_Act_THRESHOLDRELU = schema::ActivationType_THRESHOLDRELU + schema::PrimitiveType_MAX,
  Operator_Act_LINEAR = schema::ActivationType_LINEAR + schema::PrimitiveType_MAX,
  Operator_Act_HARD_TANH = schema::ActivationType_HARD_TANH + schema::PrimitiveType_MAX,
  Operator_Act_SIGN = schema::ActivationType_SIGN + schema::PrimitiveType_MAX,
  Operator_Act_SWISH = schema::ActivationType_SWISH + schema::PrimitiveType_MAX,
};

struct FusionEltwiseParameter {
  struct Node_ {
    bool is_leaf_;
    FusionEltwiseParameter *value_;  // if is_leaf_=true, value_ is a Tensor
    std::string name_;
    Node_(bool is_leaf, FusionEltwiseParameter *value, std::string value_name)
        : is_leaf_(is_leaf), value_(value), name_(std::move(value_name)) {}
  };
  OpParameter op_parameter_{"FusionEltwiseParameter", PrimitiveType_FusionEltwise, 1, schema::QuantType_QUANT_NONE};

  // Node: Duplication of extra fields of ArithmeticParameter here is aiming for FusionEltwise shape inference.
  // In one step of member method InferShape(), FusionEltwiseParameter is reinterpreted as ArithmeticParameter,
  // and below fields would be assigned in detail shape inference.
  bool broadcasting_;
  size_t ndim_;
  int activation_type_;
  int in_shape0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_elements_num0_;
  int in_shape1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_elements_num1_;

  int out_shape_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_elements_num_;

  int in_strides0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int in_strides1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int out_strides_[ARITHMETIC_SUPPORT_DIMS_NUM];

  int multiples0_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int multiples1_[ARITHMETIC_SUPPORT_DIMS_NUM];
  int eltwise_mode_;  // eltwise need

  EltwiseOperator operator_;
  std::string name_;
  std::vector<Node_> inputs_;

  FusionEltwiseParameter(EltwiseOperator operator_init, std::string kernel_name,
                         const std::vector<lite::Tensor *> &in_tensors,
                         const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {})
      : operator_(operator_init), name_(std::move(kernel_name)) {
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      auto *in_tensor = in_tensors[i];
      if (replace_map.count(in_tensor)) {
        auto *pred_param = replace_map.at(in_tensor);
        inputs_.emplace_back(false, pred_param, pred_param->name_);
        if (reinterpret_cast<void *>(in_tensor) == reinterpret_cast<void *>(pred_param)) {
          this->name_ = pred_param->name_ + "(" + this->name_ + ")";
        } else {
          this->name_ = pred_param->name_ + ", " + this->name_;
        }
      } else {
        inputs_.emplace_back(true, reinterpret_cast<FusionEltwiseParameter *>(in_tensor), "tensor" + std::to_string(i));
      }
    }
  }

  ~FusionEltwiseParameter() {
    for (const auto &input : inputs_) {
      if (!input.is_leaf_) {
        delete input.value_;
      }
    }
  }
};

constexpr EltwiseOperator Activation2Operator(ActivationType act_type) {
  return static_cast<EltwiseOperator>(act_type + schema::PrimitiveType_MAX);
}

FusionEltwiseParameter *CreateFusionEltwiseParameter(
  KernelExec *node, const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {});

bool IsEltwiseAndOperatorSupported(KernelExec *node);

class FusionEltwiseOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~FusionEltwiseOpenCLKernel() override {
    if (op_parameter_ != nullptr) {
      delete reinterpret_cast<FusionEltwiseParameter *>(op_parameter_);
      op_parameter_ = nullptr;
    }
  }

  int Prepare() override;
  int InferShape() override;
  int InitWeights() override;
  int SetGlobalLocal() override;
  int SetConstArgs() override;
  int Run() override;

  void ClearParameter() { op_parameter_ = nullptr; }

 public:
  std::string Codegen();
  std::string CodegenCore(FusionEltwiseParameter *param, const std::string &out_name = "out", int degree = 0);
  std::string GetFormatVarName(std::string name = "");
  int GetTensorIdx(lite::Tensor *in_tensor);

  static inline bool IsScalar(const std::vector<int> &shape) {
    return shape.empty() || (shape.size() == 1 && shape.front() == 1);
  }
  const std::vector<KernelExec *> *in_kernels_;

 private:
  std::map<std::string, std::string> var_names_;  // origin name -> simplified name
  const bool simplify_var_name_{true};
  std::vector<float> scalar_weights_;
  std::vector<void *> buffer_weights_;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FUSION_ELTWISE_H_
