/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/pynative/forward/do_infer.h"
#include "pipeline/pynative/base.h"
#include "utils/ms_context.h"

namespace py = pybind11;
using pybind11::literals::operator"" _a;
using PrimitivePy = mindspore::PrimitivePy;
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using BaseOpRunInfo = mindspore::pynative::BaseOpRunInfo;
using FrontendOpRunInfo = mindspore::pynative::FrontendOpRunInfo;
using InferOperation = mindspore::pynative::InferOperation;

namespace mindspore {
namespace pynative {
class TestPynativeExecute : public UT::Common {
 public:
  TestPynativeExecute() {}
};

inline ValuePtr PyAttrValue(const py::object &obj) {
  ValuePtr converted_ret;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

FrontendOpRunInfoPtr ConstructOpExecInfo() {
  py::str op_name = "Conv2D";
  py::object tensor_py_module = py::module::import("mindspore.common.tensor").attr("Tensor");
  py::object np_py_module = py::module::import("numpy");
  py::object np_ones = np_py_module.attr("ones");
  py::object np_float32 = np_py_module.attr("float32");
  py::tuple weight_dim = py::make_tuple(64, 3, 3, 3);
  py::object weight = tensor_py_module(np_float32(np_ones(weight_dim)));
  py::tuple op_params = py::make_tuple(weight);
  py::tuple inputs_dim = py::make_tuple(1, 3, 6, 6);
  py::object input = tensor_py_module(np_float32(np_ones(inputs_dim)));
  py::tuple op_inputs = py::make_tuple(input, weight);

  py::tuple kernel_size = py::make_tuple(3, 3);
  py::dict op_attrs = py::dict("out_channel"_a = 64, "kernel_size"_a = kernel_size, "mode"_a = 1, "pad_mode"_a = "same",
                               "stride"_a = 1, "dilation"_a = 1, "group"_a = 1, "data_format"_a = kOpFormat_NCHW);

  auto conv_obj = prim::GetPythonOps("conv2d_prim", "gtest_input.pynative");
  py::none py_none;
  py::args args = py::make_tuple(conv_obj, op_name, op_inputs);
  return PyNativeExecutor::GetInstance()->forward_executor()->GenerateOpRunInfo(args);
}

/// Feature: Test pynative create context
/// Description: Test pynative create context interface
/// Expectation: success
TEST_F(TestPynativeExecute, TestCreateContext) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, "CPU");
  auto ctx3 = MsContext::GetInstance();
  ASSERT_EQ(ctx3->backend_policy(), "vm");
  ASSERT_EQ(ctx3->get_param<std::string>(MS_CTX_DEVICE_TARGET), "CPU");

  ctx3->set_backend_policy("ge_only");
  ctx3->set_param<std::string>(MS_CTX_DEVICE_TARGET, "GPU");
  auto ctx4 = MsContext::GetInstance();

  ASSERT_EQ(ctx3.get(), ctx4.get());
  ASSERT_EQ(ctx4->backend_policy(), "ge_only");
  ASSERT_EQ(ctx4->get_param<std::string>(MS_CTX_DEVICE_TARGET), "GPU");
}

/// Feature: Test pynative default context
/// Description: Test pynative default context interface
/// Expectation: success
TEST_F(TestPynativeExecute, TestDefaultContext) {
  auto ctx = MsContext::GetInstance();

  ASSERT_EQ(std::string(ctx->backend_policy()), "ge_only");

  auto ctx2 = MsContext::GetInstance();

  ASSERT_EQ(ctx.get(), ctx2.get());
}

/// Feature: Test pynative infer operation
/// Description: Test pynative infer interface by using `matmul` ops
/// Expectation: success
TEST_F(TestPynativeExecute, TestInferOperator) {
  auto conv_obj = prim::GetPythonOps("matmul", "gtest_input.pynative");
  auto t1 = prim::GetPythonOps("tensor1", "gtest_input.pynative");
  auto t2 = prim::GetPythonOps("tensor2", "gtest_input.pynative");
  // Get run op info.
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = "MatMul";
  op_run_info->op_prim = conv_obj->cast<PrimitivePyPtr>();
  ASSERT_NE(op_run_info->op_prim, nullptr);
  (void)op_run_info->input_value.emplace_back(t1);
  (void)op_run_info->input_value.emplace_back(t2);
  op_run_info->input_size = 2;
  // call infer operator.
  auto infer_operator = std::make_shared<InferOperation>();
  infer_operator->DoInfer(op_run_info);
  // Check abstract.
  ASSERT_NE(op_run_info->out_value, nullptr);
  ASSERT_EQ(op_run_info->out_value->isa<ValueAny>(), true);
  auto output_abs = op_run_info->base_op_run_info.abstract;
  ASSERT_NE(output_abs, nullptr);
  ASSERT_EQ(output_abs->isa<abstract::AbstractTensor>(), true);
  auto abs_tensor = output_abs->cast<abstract::AbstractTensorPtr>();
  ASSERT_NE(abs_tensor, nullptr);
  // Check type.
  auto base_type = abs_tensor->BuildType();
  ASSERT_NE(base_type, nullptr);
  auto tensor_type = base_type->cast<TensorTypePtr>();
  ASSERT_NE(tensor_type, nullptr);
  ASSERT_EQ(tensor_type->element()->type_id(), kNumberTypeFloat32);
  // Check shape.
  auto base_shape = abs_tensor->BuildShape();
  ASSERT_NE(base_shape, nullptr);
  auto shape = base_shape->cast<abstract::ShapePtr>();
  ASSERT_NE(shape, nullptr);
  auto shape_v = shape->shape();
  ASSERT_EQ(shape_v.size(), 2);
  ASSERT_EQ(shape_v[0], 1);
  ASSERT_EQ(shape_v[1], 1);
}

}  // namespace pynative
}  // namespace mindspore
