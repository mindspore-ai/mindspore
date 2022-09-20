/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_OP_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_OP_H_

#include <utility>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>
#include "include/api/kernel.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_context.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/op_registration_factory.h"
#include "src/extendrt/delegate/tensorrt/tensor_info.h"
// #include "src/extendrt/delegate/tensorrt/cuda_impl/cublas_utils.h"
#include "src/common/log_util.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "kernel/kernel.h"
#include "include/api/types.h"
#include "mindapi/base/types.h"

namespace mindspore::lite {
constexpr int INPUT_SIZE2 = 2;
constexpr int INPUT_SIZE3 = 3;
constexpr int INPUT_SIZE4 = 4;
constexpr int INPUT_SIZE5 = 5;

struct BindingHelper {
  std::string name_;
  const void *data_{nullptr};
  nvinfer1::DataType data_type_;
  size_t size_;
  bool is_input_binding_{false};
};

struct DynamicShapeParams {
  bool support_dynamic_{true};
  bool support_hw_dynamic_{true};
};

class TensorRTRuntime;

using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;

class TensorRTOp {
 public:
  TensorRTOp(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
             const std::vector<TensorInfo> &out_tensors, std::string name);

  virtual ~TensorRTOp() = default;

  virtual int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                        const std::vector<TensorInfo> &out_tensors) = 0;

  // The weight input has been processed internally by the operator. The framework does not
  // need to process the weight input.
  virtual bool IsWeightInputHanledInner() const { return false; }

  virtual int AddInnerOp(TensorRTContext *ctx) = 0;

  virtual int SetInt8DynamicRange(TensorRTContext *ctx);

  virtual int Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine);

  const BaseOperatorPtr &GetBaseOperator();

  bool HasConst() const;

  int ReadyInputsNumber(TensorRTContext *ctx) const;

  std::string GetOpName();

  std::vector<TensorInfo> &inputs();

  ITensorHelper input(TensorRTContext *ctx, size_t i);

  ITensorHelper output(TensorRTContext *ctx, size_t i);

  std::vector<TensorInfo> &outputs();

  const std::string &type() const;

  schema::QuantType GetQuantType() const;

  void set_in_ops(const std::vector<TensorRTOp *> &in_ops);

  void set_out_ops(const std::vector<TensorRTOp *> &out_ops);

  const std::vector<TensorRTOp *> &in_ops() const;

  const std::vector<TensorRTOp *> &out_ops() const;

  void SetRuntime(TensorRTRuntime *runtime);
  cublasHandle_t GetCublasHandle() { return runtime_ ? runtime_->GetCublasHandle() : nullptr; }
  cublasLtHandle_t GetCublasLtHandle() { return runtime_ ? runtime_->GetCublasLtHandle() : nullptr; }

  DynamicShapeParams GetDynamicShapeParams() const;

  nvinfer1::ILayer *layer() { return layer_; }

  bool GetSupportInputBool();
  bool IsDynamicInput(TensorRTContext *ctx, size_t k);

  void SetSupportInputBool(bool support_input_bool);
  template <class OpsT>
  std::shared_ptr<OpsT> AsOps() {
    return std::make_shared<OpsT>(base_operator_->GetPrim());
  }

  template <class OpsT>
  static std::shared_ptr<OpsT> AsOps(const BaseOperatorPtr &base_operator) {
    return std::make_shared<OpsT>(base_operator->GetPrim());
  }

 private:
  int SetTransposeDynamicRange();

 protected:
  bool IsShapeKnown();

  nvinfer1::ILayer *layer_ = nullptr;

  nvinfer1::IShuffleLayer *transpose_layer_ = nullptr;

  BaseOperatorPtr base_operator_ = nullptr;
  std::vector<TensorInfo> in_tensors_;
  std::vector<TensorInfo> out_tensors_;

  std::vector<TensorRTOp *> in_ops_;

  std::vector<TensorRTOp *> out_ops_;

  std::string op_name_;

  std::string type_;

  schema::QuantType quant_type_ = schema::QuantType_QUANT_NONE;

  std::vector<BindingHelper> op_binding_tensor_;

  TensorRTRuntime *runtime_{nullptr};

  DynamicShapeParams dynamic_shape_params_;

  uint32_t device_id_{0};

  bool support_input_bool_{true};
};

template <class T>
TensorRTOp *GetTensorRTOp(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &inputs,
                          const std::vector<TensorInfo> &outputs, const std::string &name) {
  auto *op = new (std::nothrow) T(base_operator, inputs, outputs, name);
  if (op == nullptr) {
    MS_LOG(WARNING) << "TensorRT is nullptr.";
    return nullptr;
  }

  auto ret = op->IsSupport(base_operator, inputs, outputs);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "TensorRT op is not supported: " << name;
    delete op;
    return nullptr;
  }
  return op;
}
typedef TensorRTOp *(*TensorRTGetOp)(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &inputs,
                                     const std::vector<TensorInfo> &outputs, const std::string &name);

#define REGISTER_TENSORRT_CREATOR(KEY, TENSORRT_OP) \
  REGISTER_CLASS_CREATOR(std::string, KEY, TensorRTGetOp, GetTensorRTOp<TENSORRT_OP>);

using TensorRTRegistrationFactory = AutoRegistrationFactory<std::string, TensorRTGetOp>;
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_OP_H_
