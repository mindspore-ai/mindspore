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
#include <set>
#include <algorithm>
#include "ut/src/runtime/kernel/opencl/common.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#include "nnacl/conv_parameter.h"
#include "schema/model_v0_generated.h"

using mindspore::kernel::LiteKernel;
using mindspore::kernel::OpenCLSubGraph;
using mindspore::lite::KernelRegistry;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::lite::opencl::test {
// muti-output
void TestMain(const std::vector<ArgsTuple> &input_infos, const std::vector<ArgsTupleOut> &output_info,
              OpParameter *op_parameter, bool fp16_enable, float atol, float rtol, bool print_data) {
  std::vector<ArgsTupleWithDtype> input_infos_new;
  auto transform_fun = [](ArgsTuple in) -> ArgsTupleWithDtype {
    return ArgsTupleWithDtype(std::get<0>(in), std::get<1>(in), std::get<2>(in), kNumberTypeFloat32);
  };
  std::transform(input_infos.begin(), input_infos.end(), std::back_inserter(input_infos_new), transform_fun);
  TestMain(input_infos_new, output_info, op_parameter, fp16_enable, atol, rtol, print_data);
}

void TestMain(const std::vector<ArgsTupleWithDtype> &input_infos, const std::vector<ArgsTupleOut> &output_info,
              OpParameter *op_parameter, bool fp16_enable, float atol, float rtol, bool print_data) {
  auto primitive_type = static_cast<schema::PrimitiveType>(op_parameter->type_);
#ifdef ENABLE_V0
  static std::set<int> packed_op = {schema::v0::PrimitiveType_Conv2D, schema::v0::PrimitiveType_DeConv2D,
                                    schema::v0::PrimitiveType_DepthwiseConv2D,
                                    schema::v0::PrimitiveType_DeDepthwiseConv2D, schema::v0::PrimitiveType_MatMul};
#else
  static std::vector<int> packed_ops = {schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_Conv2dTransposeFusion,
                                        schema::PrimitiveType_MatMul};
#endif

  // simulating benchmark: session::LiteSession::CreateSession() -> session->Init()
  MS_LOG(DEBUG) << "initialize OpenCLRuntime and OpenCLAllocator";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = runtime_wrapper.GetInstance();
  ocl_runtime->SetFp16Enable(fp16_enable);
  EXPECT_TRUE(ocl_runtime->Init() == RET_OK);

  // simulating benchmark:  session_->CompileGraph() -> ConvertTensors()
  MS_LOG(DEBUG) << "create Tensors & init weight data";
  std::vector<std::shared_ptr<Tensor>> in_tensors;
  std::vector<std::shared_ptr<Tensor>> out_tensors;
  // firstly, create all Tensors
  in_tensors.reserve(input_infos.size());  // vector's capacity() is 0, so call reserve() avoiding vector re-malloc
  for (auto input_info : input_infos) {
    auto &shape = std::get<0>(input_info);
    auto category = std::get<2>(input_info);
    auto data_type = std::get<3>(input_info);
    in_tensors.emplace_back(std::make_shared<Tensor>(data_type, shape, Format_NHWC, category));
  }
  for (auto outout_info : output_info) {
    const std::vector<int> &output_shape = std::get<0>(outout_info);
    out_tensors.emplace_back(std::make_shared<Tensor>(kNumberTypeFloat32, output_shape, Format_NHWC, VAR));
  }
  // secondly, init weight Tensor's data
  std::vector<Tensor *> kernel_inputs;
  std::vector<Tensor *> subgraph_inputs;
  std::vector<Tensor *> outputs;
  std::map<Tensor *, float *> subgraph_inputs_data;
  for (int i = 0; i < in_tensors.size(); ++i) {
    auto tensor = in_tensors[i];
    auto *input_data = std::get<1>(input_infos[i]);
    kernel_inputs.push_back(tensor.get());
    if (tensor->category() != VAR) {  // tensor is weight
      // simulating src/lite_session.cc:WeightTensorNeedCopy()
      if (packed_op.count(primitive_type)) {
        tensor->set_data(input_data);
      } else {
        memcpy(tensor->MutableData(), input_data, tensor->Size());
      }
    } else {
      EXPECT_TRUE(tensor->data_type() == kNumberTypeFloat32 || tensor->data_type() == kNumberTypeInt32);
      subgraph_inputs.push_back(tensor.get());
      subgraph_inputs_data[tensor.get()] = reinterpret_cast<float *>(input_data);
    }
  }
  for (int i = 0; i < out_tensors.size(); ++i) {
    auto out_tensor = out_tensors[i];
    outputs.push_back(out_tensor.get());
  }

  // simulating benchmark:  session_->CompileGraph() -> scheduler.Schedule() -> BuildKernels()
  MS_LOG(DEBUG) << "create OpenCLKernel";
  kernel::KernelKey key{kernel::kGPU, kernel_inputs.front()->data_type(), primitive_type};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    std::cerr << "can't get registry function for: " << schema::EnumNamePrimitiveType(primitive_type)
              << ". Maybe you forget setting op_parameter_.type_ for OpParameter." << std::endl;
    free(op_parameter);
    FAIL();
  }
  auto *kernel = creator(kernel_inputs, outputs, op_parameter, nullptr, key);
  if (kernel == nullptr) {
    std::cerr << "call registry function error: " << schema::EnumNamePrimitiveType(primitive_type) << std::endl;
    free(op_parameter);
    FAIL();
  }
  kernel->set_name(schema::EnumNamesPrimitiveType()[primitive_type]);

  // simulating benchmark:  session_->CompileGraph() -> scheduler.Schedule() -> ConstructSubGraphs()
  MS_LOG(DEBUG) << "create SubGraph";
  std::vector<LiteKernel *> kernels{kernel};
  auto sub_graph = new (std::nothrow) OpenCLSubGraph(subgraph_inputs, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    return;
  }

  // call sub_graph->Init() after construct subgraph like scheduler.cc
  MS_LOG(DEBUG) << "call sub_graph->Init()";
  EXPECT_TRUE(sub_graph->Init() == RET_OK);

  // simulating benchmark:  session_->CompileGraph() -> PrepareKernels() -> OpenCLSubGraph.Prepare()
  MS_LOG(DEBUG) << "call sub_graph->Prepare()";
  EXPECT_TRUE(sub_graph->Prepare() == RET_OK);  // will set Tensor's allocator be OpenCLAllocator

  // simulating benchmark:  model->Free(), clear weight data in input_infos
  std::vector<std::unique_ptr<uint8_t[]>> saved_weights;
  for (int i = 0; i < in_tensors.size(); ++i) {
    auto &tensor = in_tensors[i];
    if (tensor->category() != VAR) {
      saved_weights.emplace_back(new uint8_t[tensor->Size()]);
      auto *weight_data = std::get<1>(input_infos[i]);
      memcpy(saved_weights.back().get(), weight_data, tensor->Size());
      srand(time(nullptr));
      memset(weight_data, rand(), tensor->Size());
    }
  }

  // simulating benchmark: LoadInput()
  MS_LOG(DEBUG) << "malloc and init input data";
  for (auto input : subgraph_inputs) {
    EXPECT_TRUE(input->MutableData() != nullptr);  // malloc Image2D & call MapBuffer()
    memcpy(input->data_c(), subgraph_inputs_data[input], input->Size());
  }

  // simulating benchmark:  MarkAccuracy() -> session_->RunGraph() -> executor_->Run() -> OpenCLSubGraph->Run()
  MS_LOG(DEBUG) << "run SubGraph & compare result";
  EXPECT_TRUE(sub_graph->Run() == RET_OK);  // will call UnmapBuffer() for input

  for (int i = 0; i < outputs.size(); ++i) {
    ocl_runtime->GetAllocator()->MapBuffer(outputs[i]->data_c(), CL_MAP_READ, nullptr, true);
    float *expect_data = reinterpret_cast<float *>(std::get<1>(output_info[i]));
    CompareOutput<float>(outputs[i]->data_c(), expect_data, outputs[i]->ElementsNum(), atol, rtol, print_data);
    ocl_runtime->GetAllocator()->UnmapBuffer(outputs[i]->data_c());
  }

  MS_LOG(DEBUG) << "release resources";
  for (auto &tensor : in_tensors) {
    if (tensor->category() != VAR && packed_op.count(primitive_type)) {
      tensor->set_data(nullptr);
    }
  }
  for (int i = 0, j = 0; i < in_tensors.size(); ++i) {  // resume weight data to input_infos
    auto &tensor = in_tensors[i];
    if (tensor->category() != VAR) {
      auto *weight_data = std::get<1>(input_infos[i]);
      memcpy(weight_data, saved_weights[j++].get(), tensor->Size());
    }
  }
  delete sub_graph;
}

// single-output
void TestMain(const std::vector<ArgsTupleWithDtype> &input_infos, std::tuple<std::vector<int>, float *> output_info,
              OpParameter *op_parameter, bool fp16_enable, float atol, float rtol, bool print_data) {
  auto primitive_type = static_cast<schema::PrimitiveType>(op_parameter->type_);
  static std::set<schema::PrimitiveType> packed_op = {
    schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_Conv2dTransposeFusion, schema::PrimitiveType_Conv2DFusion,
    schema::PrimitiveType_Conv2dTransposeFusion, schema::PrimitiveType_MatMul};

  // simulating benchmark: session::LiteSession::CreateSession() -> session->Init()
  MS_LOG(DEBUG) << "initialize OpenCLRuntime and OpenCLAllocator";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = runtime_wrapper.GetInstance();
  ocl_runtime->SetFp16Enable(fp16_enable);
  EXPECT_TRUE(ocl_runtime->Init() == RET_OK);

  // simulating benchmark:  session_->CompileGraph() -> ConvertTensors()
  MS_LOG(DEBUG) << "create Tensors & init weight data";
  std::vector<std::shared_ptr<Tensor>> tensors;
  // firstly, create all Tensors
  tensors.reserve(input_infos.size());  // vector's capacity() is 0, so call reserve() avoiding vector re-malloc
  for (auto input_info : input_infos) {
    auto &shape = std::get<0>(input_info);
    auto category = std::get<2>(input_info);
    auto data_type = std::get<3>(input_info);
    tensors.emplace_back(std::make_shared<Tensor>(data_type, shape, Format_NHWC, category));
  }
  // secondly, init weight Tensor's data
  std::vector<Tensor *> kernel_inputs;
  std::vector<Tensor *> subgraph_inputs;
  std::map<Tensor *, float *> subgraph_inputs_data;
  for (int i = 0; i < tensors.size(); ++i) {
    auto tensor = tensors[i];
    auto *input_data = std::get<1>(input_infos[i]);
    kernel_inputs.push_back(tensor.get());
    if (tensor->category() != VAR) {  // tensor is weight
      // simulating src/lite_session.cc:WeightTensorNeedCopy()
      if (packed_op.count(primitive_type)) {
        tensor->set_data(input_data);
      } else {
        memcpy(tensor->MutableData(), input_data, tensor->Size());
      }
    } else {
      EXPECT_TRUE(tensor->data_type() == kNumberTypeFloat32 || tensor->data_type() == kNumberTypeInt32);
      subgraph_inputs.push_back(tensor.get());
      subgraph_inputs_data[tensor.get()] = reinterpret_cast<float *>(input_data);
    }
  }

  const std::vector<int> &output_shape = std::get<0>(output_info);
  float *expect_data = std::get<1>(output_info);
  auto output = Tensor(kNumberTypeFloat32, output_shape, Format_NHWC, VAR);

  // simulating benchmark:  session_->CompileGraph() -> scheduler.Schedule() -> BuildKernels()
  MS_LOG(DEBUG) << "create OpenCLKernel";
  kernel::KernelKey key{kernel::kGPU, kernel_inputs.front()->data_type(), primitive_type};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    std::cerr << "can't get registry function for: " << schema::EnumNamePrimitiveType(primitive_type)
              << ". Maybe you forget setting op_parameter_.type_ for OpParameter." << std::endl;
    free(op_parameter);
    FAIL();
  }
  auto *kernel = creator(kernel_inputs, {&output}, op_parameter, nullptr, key);
  if (kernel == nullptr) {
    std::cerr << "call registry function error: " << schema::EnumNamePrimitiveType(primitive_type) << std::endl;
    free(op_parameter);
    FAIL();
  }
  kernel->set_name(schema::EnumNamesPrimitiveType()[primitive_type]);

  // simulating benchmark:  session_->CompileGraph() -> scheduler.Schedule() -> ConstructSubGraphs()
  MS_LOG(DEBUG) << "create SubGraph";
  std::vector<LiteKernel *> kernels{kernel};
  auto sub_graph = new (std::nothrow) OpenCLSubGraph(subgraph_inputs, {&output}, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    return;
  }

  // call sub_graph->Init() after construct subgraph like scheduler.cc
  MS_LOG(DEBUG) << "call sub_graph->Init()";
  EXPECT_TRUE(sub_graph->Init() == RET_OK);

  // simulating benchmark:  session_->CompileGraph() -> PrepareKernels() -> OpenCLSubGraph.Prepare()
  MS_LOG(DEBUG) << "call sub_graph->Prepare()";
  EXPECT_TRUE(sub_graph->Prepare() == RET_OK);  // will set Tensor's allocator be OpenCLAllocator

  // simulating benchmark:  model->Free(), clear weight data in input_infos
  std::vector<std::unique_ptr<uint8_t[]>> saved_weights;
  for (int i = 0; i < tensors.size(); ++i) {
    auto &tensor = tensors[i];
    if (tensor->category() != VAR) {
      saved_weights.emplace_back(new uint8_t[tensor->Size()]);
      auto *weight_data = std::get<1>(input_infos[i]);
      memcpy(saved_weights.back().get(), weight_data, tensor->Size());
      srand(time(nullptr));
      memset(weight_data, rand(), tensor->Size());
    }
  }

  // simulating benchmark: LoadInput()
  MS_LOG(DEBUG) << "malloc and init input data";
  for (auto input : subgraph_inputs) {
    EXPECT_TRUE(input->MutableData() != nullptr);  // malloc Image2D & call MapBuffer()
    memcpy(input->data_c(), subgraph_inputs_data[input], input->Size());
  }

  // simulating benchmark:  MarkAccuracy() -> session_->RunGraph() -> executor_->Run() -> OpenCLSubGraph->Run()
  MS_LOG(DEBUG) << "run SubGraph & compare result";
  EXPECT_TRUE(sub_graph->Run() == RET_OK);  // will call UnmapBuffer() for input

  // check result
  ocl_runtime->GetAllocator()->MapBuffer(output.data_c(), CL_MAP_READ, nullptr, true);
  CompareOutput<float>(output.data_c(), expect_data, output.ElementsNum(), atol, rtol, print_data);
  ocl_runtime->GetAllocator()->UnmapBuffer(output.data_c());

  MS_LOG(DEBUG) << "release resources";
  for (auto &tensor : tensors) {
    if (tensor->category() != VAR && packed_op.count(primitive_type)) {
      tensor->set_data(nullptr);
    }
  }
  for (int i = 0, j = 0; i < tensors.size(); ++i) {  // resume weight data to input_infos
    auto &tensor = tensors[i];
    if (tensor->category() != VAR) {
      auto *weight_data = std::get<1>(input_infos[i]);
      memcpy(weight_data, saved_weights[j++].get(), tensor->Size());
    }
  }
  delete sub_graph;
}

void TestMain(const std::vector<ArgsTuple> &input_infos, std::tuple<std::vector<int>, float *> output_info,
              OpParameter *op_parameter, bool fp16_enable, float atol, float rtol, bool print_data) {
  std::vector<ArgsTupleWithDtype> input_infos_new;
  auto transform_fun = [](ArgsTuple in) -> ArgsTupleWithDtype {
    return ArgsTupleWithDtype(std::get<0>(in), std::get<1>(in), std::get<2>(in), kNumberTypeFloat32);
  };
  std::transform(input_infos.begin(), input_infos.end(), std::back_inserter(input_infos_new), transform_fun);
  TestMain(input_infos_new, output_info, op_parameter, fp16_enable, atol, rtol, print_data);
}

}  // namespace mindspore::lite::opencl::test
