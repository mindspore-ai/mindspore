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
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/runtime/kernel/opencl/kernel/arithmetic.h"
#include "src/runtime/kernel/opencl/kernel/arithmetic_self.h"
#include "src/runtime/kernel/opencl/kernel/to_format.h"
#include "schema/ops_generated.h"

using mindspore::schema::ActivationType;
using mindspore::schema::PrimitiveType;
using mindspore::schema::PrimitiveType_MAX;

namespace mindspore::kernel {

constexpr schema::PrimitiveType PrimitiveType_FusionEltwise = static_cast<schema::PrimitiveType>(-100);

enum EltwiseOperator {
  // Arithmetic Primitive
  Operator_Mul = PrimitiveType_MulFusion,
  Operator_Add = PrimitiveType_AddFusion,
  Operator_Sub = PrimitiveType_SubFusion,
  Operator_Div = PrimitiveType_DivFusion,
  Operator_LogicalAnd = PrimitiveType_LogicalAnd,
  Operator_LogicalOr = PrimitiveType_LogicalOr,
  Operator_Maximum = PrimitiveType_Maximum,
  Operator_Minimum = PrimitiveType_Minimum,
  Operator_FloorDiv = PrimitiveType_FloorDiv,
  Operator_FloorMod = PrimitiveType_FloorMod,
  Operator_SquaredDifference = PrimitiveType_SquaredDifference,
  Operator_Equal = PrimitiveType_Equal,
  Operator_NotEqual = PrimitiveType_NotEqual,
  Operator_Less = PrimitiveType_Less,
  Operator_LessEqual = PrimitiveType_LessEqual,
  Operator_Greater = PrimitiveType_Greater,
  Operator_GreaterEqual = PrimitiveType_GreaterEqual,
  Operator_Eltwise = PrimitiveType_Eltwise,

  // ArithmeticSelf Primitive
  Operator_Abs = PrimitiveType_Abs,
  Operator_Ceil = PrimitiveType_Ceil,
  Operator_Cos = PrimitiveType_Cos,
  Operator_Exp = PrimitiveType_ExpFusion,
  Operator_Floor = PrimitiveType_Floor,
  Operator_Log = PrimitiveType_Log,
  Operator_LogicalNot = PrimitiveType_LogicalNot,
  Operator_Round = PrimitiveType_Round,
  Operator_Rsqrt = PrimitiveType_Rsqrt,
  Operator_Sin = PrimitiveType_Sin,
  Operator_Neg = PrimitiveType_Neg,
  Operator_Sqrt = PrimitiveType_Sqrt,
  Operator_Square = PrimitiveType_Square,

  // Other Primitive
  Operator_Scale = schema::PrimitiveType_ScaleFusion,

  // Activation
  Operator_Act_NO_ACTIVATION = schema::ActivationType_NO_ACTIVATION + PrimitiveType_MAX,
  Operator_Act_RELU = schema::ActivationType_RELU + PrimitiveType_MAX,
  Operator_Act_SIGMOID = schema::ActivationType_SIGMOID + PrimitiveType_MAX,
  Operator_Act_RELU6 = schema::ActivationType_RELU6 + PrimitiveType_MAX,
  Operator_Act_ELU = schema::ActivationType_ELU + PrimitiveType_MAX,
  Operator_Act_LEAKY_RELU = schema::ActivationType_LEAKY_RELU + PrimitiveType_MAX,
  Operator_Act_ABS = schema::ActivationType_ABS + PrimitiveType_MAX,
  Operator_Act_RELU1 = schema::ActivationType_RELU1 + PrimitiveType_MAX,
  Operator_Act_SOFTSIGN = schema::ActivationType_SOFTSIGN + PrimitiveType_MAX,
  Operator_Act_SOFTPLUS = schema::ActivationType_SOFTPLUS + PrimitiveType_MAX,
  Operator_Act_TANH = schema::ActivationType_TANH + PrimitiveType_MAX,
  Operator_Act_SELU = schema::ActivationType_SELU + PrimitiveType_MAX,
  Operator_Act_HSWISH = schema::ActivationType_HSWISH + PrimitiveType_MAX,
  Operator_Act_HSIGMOID = schema::ActivationType_HSIGMOID + PrimitiveType_MAX,
  Operator_Act_THRESHOLDRELU = schema::ActivationType_THRESHOLDRELU + PrimitiveType_MAX,
  Operator_Act_LINEAR = schema::ActivationType_LINEAR + PrimitiveType_MAX,
  Operator_Act_HARD_TANH = schema::ActivationType_HARD_TANH + PrimitiveType_MAX,
  Operator_Act_SIGN = schema::ActivationType_SIGN + PrimitiveType_MAX,
  Operator_Act_SWISH = schema::ActivationType_SWISH + PrimitiveType_MAX,
};

struct FusionEltwiseParameter {
  struct Node_ {
    bool is_leaf_;
    FusionEltwiseParameter *value_;  // if is_leaf_=true, value_ is a Tensor
    std::string name_;
    Node_(bool is_leaf, FusionEltwiseParameter *value, std::string value_name)
        : is_leaf_(is_leaf), value_(value), name_(std::move(value_name)) {}
  };
  OpParameter op_parameter_{"FusionEltwiseParameter", true, PrimitiveType_FusionEltwise, 1};
  EltwiseOperator operator_;
  std::string name_;
  std::vector<Node_> inputs_;

  FusionEltwiseParameter(EltwiseOperator operator_init, std::string kernel_name,
                         const std::vector<lite::Tensor *> &in_tensors,
                         const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {})
      : operator_(operator_init), name_(std::move(kernel_name)) {
    for (int i = 0; i < in_tensors.size(); ++i) {
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
  return static_cast<EltwiseOperator>(act_type + PrimitiveType_MAX);
}

FusionEltwiseParameter *CreateFusionEltwiseParameter(
  LiteKernel *node, const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {});

bool IsEltwiseAndOperatorSupported(LiteKernel *node);

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
  int InitWeights() override;
  void SetGlobalLocal() override;
  void SetConstArgs() override;
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

  std::map<std::string, std::string> var_names_;  // origin name -> simplified name
  const bool simplify_var_name_{true};
  std::vector<float> scalar_weights_;
  std::vector<void *> buffer_weights_;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FUSION_ELTWISE_H_
