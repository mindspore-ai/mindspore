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
#include "ir/dtype.h"
#include "transform/transform_base_test.h"
#include "common/py_func_graph_fetcher.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "operator/ops.h"
#include "transform/df_graph_manager.h"
#include "transform/convert.h"
#include "utils/utils.h"

#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif

#define private public
#include "transform/graph_runner.h"

namespace mindspore {
namespace transform {
class TestGraphRunner : public UT::Common {
 public:
  TestGraphRunner() {}
  void SetUp();
  static const std::shared_ptr<Float> kF64;
  static const std::shared_ptr<Float> kF32;

 private:
};

void TestGraphRunner::SetUp() { UT::InitPythonPath(); }
const std::shared_ptr<Float> TestGraphRunner::kF64 = std::make_shared<Float>(64);
const std::shared_ptr<Float> TestGraphRunner::kF32 = std::make_shared<Float>(32);

std::shared_ptr<DfGraphConvertor> MakeGeGraph() {
  PrimitivePtr conv2d = prim::kPrimConv2D;
  conv2d->AddAttr("stride", MakeValue(1));
  conv2d->AddAttr("pad", MakeValue(0));
  conv2d->AddAttr("pad_mode", MakeValue(std::string("pad")));
  conv2d->AddAttr("dilation", MakeValue(1));
  conv2d->AddAttr("group", MakeValue(1));
  conv2d->AddAttr("mode", MakeValue(1));
  conv2d->AddAttr("out_channel", MakeValue(2));
  conv2d->AddAttr("kernel_size", MakeValue(std::vector<int>({2, 2})));
  conv2d->AddAttr("dilation", MakeValue(1));
  conv2d->AddAttr("data_format", MakeValue(kOpFormat_NCHW));

  FuncGraphPtr anf_graph = MakeFuncGraph(conv2d, 2);
  std::shared_ptr<FuncGraphManager> ir_graph_manager = MakeManager({anf_graph});

  return std::make_shared<DfGraphConvertor>(anf_graph);
}
namespace {
std::shared_ptr<std::vector<MeTensorPtr>> DoExecGraph(const std::vector<MeTensorPtr>& inputs) {
  std::vector<GeTensorPtr> ge_tensor_ptrs = TransformUtil::ConvertInputTensors(inputs, kOpFormat_NCHW);

  std::vector<GeTensorPtr> ge_outputs;
  transform::GraphRunnerOptions options;
  transform::GraphRunner graph_runner(options);
  transform::RunOptions run_options;
  run_options.name = "fp_bp_subgraph";

  MS_LOG(INFO) << "Run func_graph begin, inputs size is: " << inputs.size();
  Status ret = graph_runner.RunGraph(run_options, ge_tensor_ptrs, &ge_outputs);
  MS_LOG(INFO) << "Run func_graph finish, outputs size is: " << ge_outputs.size();
  if (ret != Status::SUCCESS) {
    return nullptr;
  }

  std::vector<std::vector<int>> request_dims;
  std::vector<int> dims1 = {1, 1, 4, 4};
  std::vector<int> dims2 = {2, 3, 4, 5};
  std::vector<int> dims3 = {9, 9};
  request_dims.emplace_back(dims1);
  request_dims.emplace_back(dims2);
  request_dims.emplace_back(dims3);

  std::vector<MeTensorPtr> me_outputs = TransformUtil::ConvertGeTensors(ge_outputs, request_dims);

  return std::make_shared<std::vector<MeTensorPtr>>(me_outputs);
}

}  // namespace

TEST_F(TestGraphRunner, TestGeTensorConstructor) {
  // Init a data buffer
  float ge_tensor_data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};

  // Create a Tensor with wanted data type and shape
  MeTensor tensor = MeTensor(TypeId::kNumberTypeFloat32, std::vector<int>({1, 2, 3}));

  // Get the writable data pointer from the tensor
  float* me_tensor_data = reinterpret_cast<float*>(tensor.data_c(true));

  // Copy data from buffer to tensor's data
  memcpy_s(me_tensor_data, static_cast<size_t>(tensor.data().nbytes()), ge_tensor_data, sizeof(ge_tensor_data));
  PrintMeTensor(&tensor);

  std::cout << "----------------------------------" << std::endl;
  py::tuple py_tuple =
    py::make_tuple(py::make_tuple(py::make_tuple(1.1f, 2.2f, 3.3f), py::make_tuple(4.4f, 5.5f, 6.6f)));
  py::array my_arry = py::array(py_tuple).attr("astype").cast<py::function>()("float32").cast<py::array>();
  MeTensor tensor_tuple = MeTensor(my_arry, kFloat32);
  PrintMeTensor(&tensor_tuple);

  py::array tensor_array = tensor.data();
  py::array tensor_tuple_array = tensor_tuple.data();
  assert(memcmp(ge_tensor_data, tensor_array.data(), sizeof(ge_tensor_data)) == 0);
  assert(memcmp(ge_tensor_data, tensor_tuple_array.data(), sizeof(ge_tensor_data)) == 0);
}

#if (!defined ENABLE_GE)

TEST_F(TestGraphRunner, TestRunGraphException) {
  DfGraphManager& graph_manager = DfGraphManager::GetInstance();
  graph_manager.ClearGraph();

  std::map<string, MeTensorPtr> dict;
  MeTensorPtr init_tensor_ptr = MakeTensor(kF32, {2, 1, 2, 2});
  dict["x1"] = init_tensor_ptr;

  std::shared_ptr<DfGraphConvertor> convertor = MakeGeGraph();
  (*convertor).ConvertAllNode().InitParam(dict).BuildGraph();
  auto df_graph = (*convertor).GetComputeGraph();

  graph_manager.AddGraph("test_graph", df_graph);
  MeTensorPtr me_tensor_ptr = MakeTensor(kF32, {1, 1, 2, 3});

  MeTensorPtr input_ptr = MakeTensor(kF32, {1, 1, 4, 4});
  std::vector<MeTensorPtr> me_inputs;
  me_inputs.emplace_back(input_ptr);
  std::vector<MeTensorPtr> me_outputs;

  GraphRunnerOptions options;
  GraphRunner graph_runner(options);
  RunOptions run_options;
  ASSERT_TRUE(graph_runner.RunGraph(run_options, me_inputs, &me_outputs) != Status::SUCCESS);
  run_options.name = "test_graph";
  ASSERT_TRUE(graph_runner.RunGraph(run_options, me_inputs, &me_outputs) == Status::SUCCESS);

  GraphRunner graph_runner2(options);
  ASSERT_TRUE(graph_runner2.RunGraph(run_options, me_inputs, &me_outputs) == Status::SUCCESS);

  // when the GraphManager is empty
  graph_manager.ClearGraph();
  GraphRunner graph_runner3(options);
  ASSERT_TRUE(graph_runner3.RunGraph(run_options, me_inputs, &me_outputs) != Status::SUCCESS);
}

TEST_F(TestGraphRunner, TestRunGraph) {
  DfGraphManager& graph_manager = DfGraphManager::GetInstance();
  graph_manager.ClearGraph();

  std::shared_ptr<DfGraphConvertor> convertor = MakeGeGraph();
  std::map<std::string, MeTensorPtr> dict;
  dict.emplace("x1", MakeTensor(kF32, {2, 1, 2, 2}));

  (*convertor).ConvertAllNode().InitParam(dict).BuildGraph();
  graph_manager.AddGraph("test_graph", (*convertor).GetComputeGraph());

  TypePtr type_id = kFloat32;

  py::tuple tuple = py::make_tuple(
    py::make_tuple(py::make_tuple(py::make_tuple(1.0, 2.0, 3.0, 4.0), py::make_tuple(4.0, 5.0, 6.0, 7.0))),
    py::make_tuple(py::make_tuple(py::make_tuple(1.0, 2.0, 3.0, 4.0), py::make_tuple(4.0, 5.0, 6.0, 7.0))));
  py::array array = py::array(tuple);
  MeTensorPtr me_tensor_ptr = std::make_shared<MeTensor>(array, type_id);

  MS_LOG(INFO) << "inputs me tensor data is: ";
  PrintMeTensor(&(*me_tensor_ptr));

  std::vector<MeTensorPtr> me_inputs;
  me_inputs.emplace_back(me_tensor_ptr);
  std::vector<MeTensorPtr> me_outputs;

  GraphRunnerOptions options;
  GraphRunner graph_runner(options);
  RunOptions run_options;
  run_options.name = "test_graph";
  ASSERT_TRUE(graph_runner.RunGraph(run_options, me_inputs, &me_outputs) == Status::SUCCESS);
  MS_LOG(INFO) << "outputs me tensor data is: ";
  for (auto i = 0; i < me_outputs.size(); i++) {
    PrintMeTensor(&(*me_outputs[i]));
  }
}

TEST_F(TestGraphRunner, TestAPI) {
  DfGraphManager& graph_manager = DfGraphManager::GetInstance();
  graph_manager.ClearGraph();

  std::shared_ptr<DfGraphConvertor> convertor = MakeGeGraph();
  std::map<std::string, MeTensorPtr> dict;
  dict.emplace("x1", MakeTensor(kF32, {2, 1, 2, 2}));

  (*convertor).ConvertAllNode().InitParam(dict).BuildGraph();
  (*convertor).DrawComputeGraph("TestGraphRunner_TestAPI_Training.dot");
  graph_manager.AddGraph("fp_bp_subgraph", (*convertor).GetComputeGraph());

  MeTensorPtr input_ptr1 = MakeTensor(kF32, {1, 1, 4, 4});
  MeTensorPtr input_ptr2 = MakeTensor(kF32, {2, 3, 4, 5});
  MeTensorPtr input_ptr3 = MakeTensor(kF32, {9, 9, 1, 1});
  std::vector<MeTensorPtr> me_inputs;
  std::vector<MeTensorPtr> me_outputs;
  me_inputs.emplace_back(input_ptr1);
  me_inputs.emplace_back(input_ptr2);
  me_inputs.emplace_back(input_ptr3);

  auto ret = DoExecGraph(me_inputs);

  ASSERT_TRUE(ret != nullptr);

  me_outputs = *ret;
  MS_LOG(INFO) << "outputs me tensor data is: ";
  for (auto tensor : me_outputs) {
    PrintMeTensor(&(*tensor));
  }
}
#endif

}  // namespace transform
}  // namespace mindspore
