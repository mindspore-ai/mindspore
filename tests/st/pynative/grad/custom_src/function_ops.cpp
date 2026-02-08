/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <string>
#include "utils/stream_guard.h"
#include "custom_op_api.h"

namespace mindspore::pynative {
namespace autograd {
ShapeVector BroadcastInferShape(const TensorPtr &t1, const TensorPtr &t2) {
  ShapeVector s1 = t1->shape();
  ShapeVector s2 = t2->shape();
  ShapeVector out_shape(std::max(s1.size(), s2.size()), 1LL);
  if (out_shape.empty()) {
    return out_shape;
  }
  for (size_t i = out_shape.size(); i > 0; i--) {
    if (i <= s1.size() && s1[s1.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s1[s1.size() - i];
    } else if (i <= s2.size() && s2[s2.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s2[s2.size() - i];
    }
  }
  return out_shape;
}

class CustomAdd : public Function<CustomAdd> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto output = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnAdd", {x, y, MakeValue<int64_t>(1)}, {output});
    return output;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, dout};
  }
};

class CustomIndex : public Function<CustomIndex> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr x, const TensorPtrList &index) {
    auto output = std::make_shared<Tensor>(x->data_type(), ShapeVector({1}));
    std::vector<ValuePtr> values;
    for (const auto &item : index) {
      (void)values.emplace_back(item->cast<ValuePtr>());
    }
    auto tuple_index = std::make_shared<ValueTuple>(values);
    custom::CustomLaunchAclnn("aclnnIndex", {x, tuple_index}, {output});
    return output;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, nullptr};
  }
};

class CustomBroadcastTo : public Function<CustomBroadcastTo> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr x, const std::vector<int64_t> &shape) {
    auto output = std::make_shared<Tensor>(x->data_type(), shape);
    std::vector<ValuePtr> values;
    for (const auto &item : shape) {
      (void)values.emplace_back(MakeValue<int64_t>(item));
    }
    auto tuple_index = std::make_shared<ValueTuple>(values);
    custom::CustomLaunchAclnn("aclnnExpand", {x, tuple_index}, {output});
    return output;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];
    return {dout, nullptr};
  }
};

class CustomMul : public Function<CustomMul> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto output = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});
    bool x_require_grad = ctx->NeedGrad(x);
    bool y_require_grad = ctx->NeedGrad(y);
    if (x_require_grad || y_require_grad) {
      ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
    }
    return output;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];

    TensorPtr grad_x = nullptr;
    TensorPtr grad_y = nullptr;

    if (ctx->NeedsInputGrad(0)) {
      grad_x = std::make_shared<Tensor>(dout->data_type(), BroadcastInferShape(dout, saved[0]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[0]}, {grad_x});
    }
    if (ctx->NeedsInputGrad(1)) {
      grad_y = std::make_shared<Tensor>(dout->data_type(), BroadcastInferShape(dout, saved[1]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[1]}, {grad_y});
    }

    return {grad_x, grad_y};
  }
};

class CustomMulNoGradOutput : public Function<CustomMulNoGradOutput> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto out = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {out});
    ctx->MarkNonDifferentiable({out});
    return out;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) {
    return {grad_output[0], grad_output[0]};
  }
};

class CustomMulNoGradInput : public Function<CustomMulNoGradInput> {
 public:
  static TensorPtrList Forward(AutogradContext *ctx, const TensorPtr &input) {
    auto out = std::make_shared<Tensor>(input->data_type(), input->shape());
    custom::CustomLaunchAclnn("aclnnMul", {input, input}, {out});
    ctx->MarkNonDifferentiable({input});
    return {input, out};
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) {
    return {grad_output[0], ctx->NeedsInputGrad(0) ? grad_output[0] : nullptr};
  }
};

class CustomInplaceMulOp : public Function<CustomInplaceMulOp> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {x});
    ctx->MarkDirty({x});
    if (ctx->NeedGrad(x)) {
      ctx->SaveForBackward({y});
    }
    return x;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) {
    TensorPtr grad_input = nullptr;
    if (ctx->NeedsInputGrad(0)) {
      auto saved_tensor = ctx->GetSavedTensors();
      grad_input =
        std::make_shared<Tensor>(grad_output[0]->data_type(), BroadcastInferShape(grad_output[0], saved_tensor[0]));
      custom::CustomLaunchAclnn("aclnnMul", {grad_output[0], saved_tensor[0]}, {grad_input});
    }
    return {grad_input};
  }
};

class CustomMulPyboost : public Function<CustomMulPyboost> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto output = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));

    auto p = std::make_shared<Primitive>("CustomLaunchAclnn");
    auto op =
      std::make_shared<kernel::pyboost::OpRunner>(p, runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend));
    op->set_stream_id(CurrentStream::id());
    op->set_outputs({output});
    // No need to convert input
    kernel::pyboost::PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, y);
    kernel::pyboost::PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

    // Async
    kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x, y]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      kernel::pyboost::PyBoostUtils::MallocOpInputs(device_context, x, y);
      // Malloc for output tensors
      kernel::pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnMul, device_context, op->stream_id(), x, y, outputs[0]);
      MS_LOG(DEBUG) << "Run device task Add end";
    }));

    bool x_require_grad = ctx->NeedGrad(x);
    bool y_require_grad = ctx->NeedGrad(y);
    if (x_require_grad || y_require_grad) {
      ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
    }
    return output;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];

    TensorPtr grad_x = nullptr;
    TensorPtr grad_y = nullptr;

    if (ctx->NeedsInputGrad(0)) {
      grad_x = std::make_shared<Tensor>(dout->data_type(), BroadcastInferShape(dout, saved[0]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[0]}, {grad_x});
    }
    if (ctx->NeedsInputGrad(1)) {
      grad_y = std::make_shared<Tensor>(dout->data_type(), BroadcastInferShape(dout, saved[1]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[1]}, {grad_y});
    }

    return {grad_x, grad_y};
  }
};

class CustomMulReturnIncorrectGradNum : public Function<CustomMulReturnIncorrectGradNum> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto out = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {out});
    return out;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) { return {grad_output[0]}; }
};

class CustomMulsReturnGradError : public Function<CustomMulsReturnGradError> {
 public:
  static TensorPtr Forward(AutogradContext *ctx, const TensorPtr &x, const int64_t n) {
    auto out = std::make_shared<Tensor>(x->data_type(), x->shape());
    custom::CustomLaunchAclnn("aclnnMuls", {x, MakeValue<int64_t>(n)}, {out});
    return out;
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) {
    return {grad_output[0], grad_output[0]};
  }
};

class CustomMultiOutputOp : public Function<CustomMultiOutputOp> {
 public:
  static TensorPtrList Forward(AutogradContext *ctx, const TensorPtr &x, const TensorPtr &y) {
    auto out1 = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {out1});
    auto out2 = std::make_shared<Tensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnAdd", {x, y, MakeValue<int64_t>(1)}, {out2});
    ctx->SetMaterializeGrads(true);
    return {out1, out2};
  }

  static TensorPtrList Backward(AutogradContext *ctx, TensorPtrList grad_output) {
    return {grad_output[0], grad_output[1]};
  }
};

TensorPtr run_custom_add(const tensor::TensorPtr &x, const tensor::TensorPtr &y) { return CustomAdd::Apply(x, y); }

TensorPtr run_custom_index(const TensorPtr x, const TensorPtrList &index) { return CustomIndex::Apply(x, index); }

TensorPtr run_custom_broadcast_to(const TensorPtr x, const std::vector<int64_t> &index) {
  return CustomBroadcastTo::Apply(x, index);
}

TensorPtr run_custom_mul(const tensor::TensorPtr &x, const tensor::TensorPtr &y) { return CustomMul::Apply(x, y); }

TensorPtr run_inplace_mul_op(const tensor::TensorPtr &input, const tensor::TensorPtr &other) {
  return CustomInplaceMulOp::Apply(input, other);
}

TensorPtr run_mul_mark_no_diff_output(const tensor::TensorPtr &input, const tensor::TensorPtr &other) {
  return CustomMulNoGradOutput::Apply(input, other);
}

TensorPtrList run_mul_mark_no_diff_input(const tensor::TensorPtr &input) { return CustomMulNoGradInput::Apply(input); }

TensorPtr run_custom_mul_pyboost(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  return CustomMulPyboost::Apply(x, y);
}

TensorPtr run_custom_mul_return_incorrect_grad_num(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  return CustomMulReturnIncorrectGradNum::Apply(x, y);
}

TensorPtr run_custom_muls_return_grad_error(const tensor::TensorPtr &x, const int64_t n) {
  return CustomMulsReturnGradError::Apply(x, n);
}

TensorPtrList run_custom_multi_output_op(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  return CustomMultiOutputOp::Apply(x, y);
}
}  // namespace autograd
}  // namespace mindspore::pynative

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("add", &mindspore::pynative::autograd::run_custom_add, "out = x + y");
  m.def("index", &mindspore::pynative::autograd::run_custom_index, "tensor[index]");
  m.def("broadcast_to", &mindspore::pynative::autograd::run_custom_broadcast_to, "shape1->shape2");
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "out = x * y");
  m.def("inplace_mul", &mindspore::pynative::autograd::run_inplace_mul_op, "x = x * y");
  m.def("mul_no_diff_out", &mindspore::pynative::autograd::run_mul_mark_no_diff_output, "out = x * y, no diff output");
  m.def("mul_no_diff_in", &mindspore::pynative::autograd::run_mul_mark_no_diff_input, "out = x * y, no diff input");
  m.def("pyboost_mul", &mindspore::pynative::autograd::run_custom_mul_pyboost, "out = x * y");
  m.def("mul_return_incorrect_grad_num", &mindspore::pynative::autograd::run_custom_mul_return_incorrect_grad_num,
        "custom mul backward return incorrect grad num");
  m.def("muls_return_grad_error", &mindspore::pynative::autograd::run_custom_muls_return_grad_error,
        "custom muls backward return error grad");
  m.def("multi_output_op", &mindspore::pynative::autograd::run_custom_multi_output_op, "multi output op");
}
