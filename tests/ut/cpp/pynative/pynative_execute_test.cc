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
#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "pipeline/parse/python_adapter.h"
#include "pipeline/parse/data_converter.h"
#include "operator/ops.h"
#include "pynative/pynative_execute.h"
#include "utils/context/ms_context.h"
#include "utils/utils.h"

namespace py = pybind11;
using pybind11::literals::operator"" _a;
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;

namespace mindspore {
namespace pynative {
class TestPynativeExecute : public UT::Common {
 public:
  TestPynativeExecute() {}
};

inline ValuePtr PyAttrValue(const py::object& obj) {
  ValuePtr converted_ret;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

OpExecInfoPtr ConstructOpExecInfo() {
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
  return GenerateOpExecInfo(py::make_tuple(conv_obj, op_name, op_inputs));
}

TEST_F(TestPynativeExecute, TestRunOpInVM) {
  py::tuple result;
  PynativeStatusCode status;
  auto op_exec_info_ptr = ConstructOpExecInfo();
  result = pynative::RunOpInVM(op_exec_info_ptr, &status);
  ASSERT_EQ(status, PYNATIVE_SUCCESS);
}

TEST_F(TestPynativeExecute, TestRunOp) {
  py::none py_none;
  auto op_exec_info_ptr = ConstructOpExecInfo();
  py::tuple outputs = pynative::RunOp(py::make_tuple(op_exec_info_ptr->py_primitive, op_exec_info_ptr->op_name,
                                                     op_exec_info_ptr->op_inputs));
  if (outputs.size() == 0) {
    FAIL();
  } else {
    SUCCEED();
  }
}

TEST_F(TestPynativeExecute, TestCreateContext) {
  auto ctx3 = MsContext::GetInstance();
  ASSERT_EQ(ctx3->backend_policy(), "vm");
  ASSERT_EQ(ctx3->device_target(), "CPU");

  ctx3->set_backend_policy("ge_only");
  ctx3->set_device_target("GPU");
  auto ctx4 = MsContext::GetInstance();

  ASSERT_EQ(ctx3.get(), ctx4.get());
  ASSERT_EQ(ctx4->backend_policy(), "ge_only");
  ASSERT_EQ(ctx4->device_target(), "GPU");
}

TEST_F(TestPynativeExecute, TestDefaultContext) {
  auto ctx = MsContext::GetInstance();

  ASSERT_EQ(std::string(ctx->backend_policy()), "ge_only");

  auto ctx2 = MsContext::GetInstance();

  ASSERT_EQ(ctx.get(), ctx2.get());
}

}  // namespace pynative
}  // namespace mindspore
