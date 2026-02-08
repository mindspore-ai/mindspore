/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include <map>
#include <string>
#include "pybind11/pybind11.h"
#include "include/utils/callbacks.h"
#include "common/common_test.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "frontend/jit/ps/pipeline.h"
#include "include/utils/python_adapter.h"
#include "mindspore/ccsrc/backend/ge_backend/graph_ir/df_graph_manager.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#ifdef ENABLE_D
#include "mindspore/ccsrc/backend/ge_backend/graph_ir/callbacks_ge.h"
#endif

namespace mindspore {
class TestCallback : public UT::Common {
 public:
  TestCallback() {}
};

/*
 * # ut and python static info not share
TEST_F(TestCallback, test_get_anf_tensor_shape) {
  py::object obj = python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_class", "test_get_object_graph");
  FuncGraphPtr func_graph = pipeline::GraphExecutorPy::GetInstance()->GetFuncGraphPy(obj);
  backend::ge_backend::DfGraphManager::GetInstance().SetAnfGraph(func_graph);
  std::shared_ptr<std::vector<int64_t>> param_shape_ptr = std::make_shared<std::vector<int64_t>>();
  bool get_shape = callbacks::GetParameterShape(func_graph, "weight", param_shape_ptr);
  ASSERT_TRUE(get_shape == true);
}

TEST_F(TestCallback, test_checkpoint_save_op) {
  py::object obj = python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_class", "test_get_object_graph");
  FuncGraphPtr func_graph = pipeline::GraphExecutorPy::GetInstance()->GetFuncGraphPy(obj);
  backend::ge_backend::DfGraphManager::GetInstance().SetAnfGraph(func_graph);

#define DTYPE float
  ge::DataType dt = ge::DataType::DT_FLOAT;

  std::vector<float> data1 = {1.1, 2.2, 3.3, 4.4, 6.6, 7.7, 8.8, 9.9};
  auto data = data1;
  ge::Shape shape({2, 2, 2, 1});
  ge::Format format = ge::Format::FORMAT_NCHW;
  ge::TensorDesc desc(shape, format, dt);
  backend::ge_backend::GeTensorPtr ge_tensor_ptr =
    std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(DTYPE));
  std::map<std::string, GeTensor> param_map;
  param_map.insert(std::pair<std::string, GeTensor>("weight", *ge_tensor_ptr));
  param_map.insert(std::pair<std::string, GeTensor>("network.weight", *ge_tensor_ptr));
  int ret = callbacks::CheckpointSaveCallback(0, param_map);
MS_LOG(INFO) << "ret=" << ret;
  ASSERT_EQ(ret, 0);
}
*/

/*
TEST_F(TestCallback, test_summary_save_op) {
    py::object obj = python_adapter::CallPyFn(
            "gtest_input.pipeline.parse.parse_class", "test_get_object_graph");
    FuncGraphPtr func_graph = obj.cast<FuncGraphPtr>();
    backend::ge_backend::DfGraphManager::GetInstance().SetAnfGraph(func_graph);

    #define DTYPE float
    ge::DataType dt = ge::DataType::DT_FLOAT;

    float data1 = 1.1;
    float data2 = 2.1;
    ge::Shape shape({1, 1, 1, 1});
    ge::Format format = ge::Format::FORMAT_NCHW;
    ge::TensorDesc desc(shape, format, dt);
    GeTensorPtr ge_tensor_ptr1 = std::make_shared<GeTensor>(desc,
                                                            reinterpret_cast<uint8_t *>(&data1),
                                                            sizeof(DTYPE));
    GeTensorPtr ge_tensor_ptr2 = std::make_shared<GeTensor>(desc,
                                                            reinterpret_cast<uint8_t *>(&data2),
                                                            sizeof(DTYPE));
    std::map<std::string, GeTensor> param_map;
    param_map.insert(std::pair<std::string, GeTensor>("x1[:Scalar]", *ge_tensor_ptr1));
    param_map.insert(std::pair<std::string, GeTensor>("x2[:Scalar]", *ge_tensor_ptr2));
    int ret = callbacks::SummarySaveCallback(0, param_map);
MS_LOG(INFO) << "ret=" << ret;
    ASSERT_TRUE(ret == 0);
}
*/
}  // namespace mindspore
