/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_

#include <utility>
#include <NvInfer.h>
#include <string>
#include <vector>
#include "include/api/kernel.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
constexpr int INPUT_SIZE2 = 2;
constexpr int INPUT_SIZE3 = 3;
constexpr int INPUT_SIZE4 = 4;

struct ITensorHelper {
  nvinfer1::ITensor *trt_tensor_{nullptr};
  mindspore::Format format_;
};

class TensorRTOp {
 public:
  explicit TensorRTOp(const schema::Primitive *primitive, std::vector<mindspore::MSTensor> in_tensors,
                      std::vector<mindspore::MSTensor> out_tensors, std::string name)
      : op_primitive_(primitive),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        op_name_(std::move(name)) {
    if (primitive != nullptr) {
      this->type_ = primitive->value_type();
    }
  }

  virtual ~TensorRTOp() = default;

  virtual int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                        const std::vector<mindspore::MSTensor> &out_tensors) = 0;

  virtual int AddInnerOp(nvinfer1::INetworkDefinition *network) = 0;

  const schema::Primitive *GetPrimitive();

  void AddInnerInTensors(ITensorHelper tensor);

  void AddInnerOutTensors(ITensorHelper tensor);

  std::vector<ITensorHelper> &GetInnerOutTensor();

  std::vector<ITensorHelper> &GetInnerInTensors();

  std::string GetOpName();

  std::vector<mindspore::MSTensor> &inputs();

  std::vector<mindspore::MSTensor> &outputs();

  schema::PrimitiveType type() const;

  void set_in_ops(const std::vector<TensorRTOp *> &in_ops);

  void set_out_ops(const std::vector<TensorRTOp *> &out_ops);

  const std::vector<TensorRTOp *> &in_ops() const;

  const std::vector<TensorRTOp *> &out_ops() const;

 protected:
  bool IsShapeKnown();

  std::vector<nvinfer1::ILayer *> layers_;

  const schema::Primitive *op_primitive_;

  std::vector<mindspore::MSTensor> in_tensors_;

  std::vector<mindspore::MSTensor> out_tensors_;

  std::vector<ITensorHelper> tensorrt_in_tensors_;

  std::vector<ITensorHelper> tensorrt_out_tensors_;

  std::vector<TensorRTOp *> in_ops_;

  std::vector<TensorRTOp *> out_ops_;

  std::string op_name_;

  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
};

template <class T>
TensorRTOp *GetTensorRTOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name) {
  auto *op = new (std::nothrow) T(primitive, in_tensors, out_tensors, name);
  if (op == nullptr) {
    MS_LOG(ERROR) << "TensorRT is nullptr.";
    return nullptr;
  }

  auto ret = op->IsSupport(primitive, in_tensors, out_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRT op is not supported.";
    delete op;
    return nullptr;
  }
  return op;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_
