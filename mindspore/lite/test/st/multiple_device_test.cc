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

#include "common/common_test.h"
#include "schema/inner/model_generated.h"
#include "src/litert/lite_session.h"
#include "src/executor/sub_graph_kernel.h"
#include "ir/dtype/type_id.h"
#include "include/api/model.h"
#include "src/litert/cxx_api/converters.h"
#include "src/litert/cxx_api/model/model_impl.h"

using mindspore::kernel::KernelExec;
using mindspore::kernel::KernelKey;
using mindspore::lite::InnerContext;
using mindspore::lite::LiteSession;
using mindspore::lite::Tensor;

class MultipleDeviceTest : public mindspore::CommonTest {
 public:
  MultipleDeviceTest() = default;
};

void CreateMultyModel1(mindspore::schema::MetaGraphT *meta_graph) {
  meta_graph->name = "graph";
  meta_graph->version = mindspore::Version();

  /* CPU GPU NPU support*/
  auto cos = std::make_unique<mindspore::schema::CNodeT>();
  cos->inputIndex = {0};
  cos->outputIndex = {1};
  cos->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cos->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cos_primitive = new mindspore::schema::CosT;
  cos->primitive->value.value = cos_primitive;
  cos->name = "cos";

  /* CPU GPU support */
  auto exp = std::make_unique<mindspore::schema::CNodeT>();
  exp->inputIndex = {1};
  exp->outputIndex = {2};
  exp->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  exp->primitive->value.type = mindspore::schema::PrimitiveType_ExpFusion;
  auto exp_primitive = new mindspore::schema::ExpFusionT;
  exp->primitive->value.value = exp_primitive;
  exp->name = "exp";

  /* CPU support */
  auto where = std::make_unique<mindspore::schema::CNodeT>();
  where->inputIndex = {2, 5, 6};
  where->outputIndex = {3};
  where->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  where->primitive->value.type = mindspore::schema::PrimitiveType_Where;
  auto where_primitive = new mindspore::schema::WhereT;
  where->primitive->value.value = where_primitive;
  where->name = "where";

  /* CPU NPU GPU support */
  auto sin = std::make_unique<mindspore::schema::CNodeT>();
  sin->inputIndex = {3};
  sin->outputIndex = {4};
  sin->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  sin->primitive->value.type = mindspore::schema::PrimitiveType_Sin;
  auto sin_primitive = new mindspore::schema::SinT;
  sin->primitive->value.value = sin_primitive;
  sin->name = "sin";

  /* tensors */
  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_Parameter;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {1, 1, 1, 1};
  tensor0->offset = -1;
  tensor0->name = "tensor0";

  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_Parameter;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {1, 1, 1, 1};
  tensor1->offset = -1;
  tensor1->name = "tensor1";

  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_Parameter;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {1, 1, 1, 1};
  tensor2->offset = -1;
  tensor2->name = "tensor2";

  auto tensor3 = std::make_unique<mindspore::schema::TensorT>();
  tensor3->nodeType = mindspore::lite::NodeType_Parameter;
  tensor3->format = mindspore::schema::Format_NHWC;
  tensor3->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor3->dims = {1, 2, 2, 1};
  tensor3->offset = -1;
  tensor3->name = "tensor3";

  auto tensor4 = std::make_unique<mindspore::schema::TensorT>();
  tensor4->nodeType = mindspore::lite::NodeType_Parameter;
  tensor4->format = mindspore::schema::Format_NHWC;
  tensor4->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor4->dims = {1, 2, 2, 1};
  tensor4->offset = -1;
  tensor4->name = "tensor4";

  auto tensor5 = std::make_unique<mindspore::schema::TensorT>();
  tensor5->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor5->format = mindspore::schema::Format_NHWC;
  tensor5->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor5->data.resize(4 * sizeof(float));
  std::vector<float> t5 = {1.0, 2.0, 3.0, 4.0};
  memcpy(tensor5->data.data(), t5.data(), 4 * sizeof(float));
  tensor5->data.resize(4 * sizeof(float));
  tensor5->dims = {1, 2, 2, 1};
  tensor5->offset = -1;
  tensor5->name = "tensor5";

  auto tensor6 = std::make_unique<mindspore::schema::TensorT>();
  tensor6->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor6->format = mindspore::schema::Format_NHWC;
  tensor6->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor6->data.resize(4 * sizeof(float));
  std::vector<float> t6 = {1.0, 2.0, 3.0, 4.0};
  memcpy(tensor6->data.data(), t6.data(), 4 * sizeof(float));
  tensor6->dims = {1, 2, 2, 1};
  tensor6->offset = -1;
  tensor6->name = "tensor6";

  meta_graph->nodes.emplace_back(std::move(cos));
  meta_graph->nodes.emplace_back(std::move(exp));
  meta_graph->nodes.emplace_back(std::move(where));
  meta_graph->nodes.emplace_back(std::move(sin));

  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));
  meta_graph->allTensors.emplace_back(std::move(tensor3));
  meta_graph->allTensors.emplace_back(std::move(tensor4));
  meta_graph->allTensors.emplace_back(std::move(tensor5));
  meta_graph->allTensors.emplace_back(std::move(tensor6));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {4};
}

void CreateMultyModel2(mindspore::schema::MetaGraphT *meta_graph) {
  meta_graph->name = "graph";

  /* CPU GPU NPU support*/
  auto cos = std::make_unique<mindspore::schema::CNodeT>();
  cos->inputIndex = {0};
  cos->outputIndex = {1};
  cos->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cos->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cos_primitive = new mindspore::schema::CosT;
  cos->primitive->value.value = cos_primitive;
  cos->name = "cos";

  /* CPU GPU support */
  auto exp = std::make_unique<mindspore::schema::CNodeT>();
  exp->inputIndex = {1};
  exp->outputIndex = {2};
  exp->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  exp->primitive->value.type = mindspore::schema::PrimitiveType_ExpFusion;
  auto exp_primitive = new mindspore::schema::ExpFusionT;
  exp->primitive->value.value = exp_primitive;
  exp->name = "exp";

  /* tensors */
  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_Parameter;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {1, 2, 2, 1};
  tensor0->offset = -1;
  tensor0->name = "tensor0";

  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_Parameter;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {1, 2, 2, 1};
  tensor1->offset = -1;
  tensor1->name = "tensor1";

  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_Parameter;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {1, 2, 2, 1};
  tensor2->offset = -1;
  tensor2->name = "tensor2";

  meta_graph->nodes.emplace_back(std::move(cos));
  meta_graph->nodes.emplace_back(std::move(exp));

  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};
}

enum MultyDeviceMode1 { CPU, NPU, GPU, CPU_GPU, GPU_CPU, NPU_CPU, NPU_GPU_CPU, NPU2, GPU_NPU2 };
void CheckResult(std::vector<mindspore::kernel::KernelExec *> kernels, int mode) {
  /*
   *          cos     exp   where   sin
   * CPU       *       *      *      *
   * GPU       *       *             *
   * NPU       *       *             *
   *
   * */

  if (mode == CPU) {
    ASSERT_EQ(1, kernels.size());
    /* CPU : cos exp where sin */
    auto subgraph1 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(0));
    ASSERT_EQ(4, subgraph1->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kCPU, subgraph1->desc().arch);

  } else if (mode == NPU_CPU) {
    ASSERT_EQ(3, kernels.size());
    /* NPU : cos exp */
    auto subgraph0 = kernels.at(0);
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kDelegate, subgraph0->desc().arch);
    /* CPU : where */
    auto subgraph1 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(1));
    ASSERT_EQ(1, subgraph1->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kCPU, subgraph1->desc().arch);
    /* NPU : sin */
    auto subgraph2 = kernels.at(2);
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kDelegate, subgraph2->desc().arch);

  } else if (mode == GPU_CPU) {
    /* GPU >  CPU */
    ASSERT_EQ(3, kernels.size());
    /* GPU : to_format cos exp to_format */
    auto subgraph0 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(0));
    ASSERT_EQ(2 + 2, subgraph0->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kGPU, subgraph0->desc().arch);
    /* CPU : where */
    auto subgraph1 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(1));
    ASSERT_EQ(1, subgraph1->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kCPU, subgraph1->desc().arch);
    /* GPU : to_format sin to_format */
    auto subgraph2 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(2));
    ASSERT_EQ(3, subgraph2->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kGPU, subgraph2->desc().arch);

  } else if (mode == NPU_GPU_CPU) {
    /* NPU > GPU >  CPU */
    ASSERT_EQ(3, kernels.size());
    /* NPU : cos exp */
    auto subgraph1 = kernels.at(0);
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kDelegate, subgraph1->desc().arch);
    /* CPU : where */
    auto subgraph2 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(1));
    ASSERT_EQ(1, subgraph2->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kCPU, subgraph2->desc().arch);
    /* NPU : sin */
    auto subgraph3 = kernels.at(2);
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kDelegate, subgraph3->desc().arch);
  } else if (mode == NPU2) {
    /* NPU > GPU */
    ASSERT_EQ(1, kernels.size());
    /* NPU : cos exp */
    auto subgraph0 = kernels.at(0);
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kDelegate, subgraph0->desc().arch);
  } else if (mode == GPU_NPU2) {
    /* GPU > NPU */
    ASSERT_EQ(1, kernels.size());
    /* GPU : to_format cos exp to_format */
    auto subgraph1 = reinterpret_cast<mindspore::kernel::SubGraphKernel *>(kernels.at(0));
    ASSERT_EQ(4, subgraph1->nodes().size());
    ASSERT_EQ(mindspore::kernel::KERNEL_ARCH::kGPU, subgraph1->desc().arch);
  }
}

TEST_F(MultipleDeviceTest, NewApi1) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateMultyModel1(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();

  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto context = std::shared_ptr<mindspore::Context>(new mindspore::Context());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());

  mindspore::Model *model = new mindspore::Model();
  auto ret = model->Build(content, size, mindspore::kMindIR_Lite, context);
  ASSERT_EQ(false, ret.IsOk());

  delete model;
}

TEST_F(MultipleDeviceTest, NewApi2) {
  mindspore::Context context;
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());

  auto inner_context = std::shared_ptr<mindspore::lite::InnerContext>(mindspore::ContextUtils::Convert(&context));
  ASSERT_EQ(inner_context->device_list_.size(), 3);
  ASSERT_EQ(inner_context->device_list_.at(0).device_type_, mindspore::lite::DT_NPU);
  ASSERT_EQ(inner_context->device_list_.at(1).device_type_, mindspore::lite::DT_CPU);
  ASSERT_EQ(inner_context->device_list_.at(2).device_type_, mindspore::lite::DT_GPU);
}

TEST_F(MultipleDeviceTest, NewApi3) {
  mindspore::Context context;
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());

  auto inner_context = std::shared_ptr<mindspore::lite::InnerContext>(mindspore::ContextUtils::Convert(&context));
  ASSERT_EQ(inner_context->device_list_.size(), 2);
  ASSERT_EQ(inner_context->device_list_.at(0).device_type_, mindspore::lite::DT_CPU);
  ASSERT_EQ(inner_context->device_list_.at(1).device_type_, mindspore::lite::DT_NPU);
}

TEST_F(MultipleDeviceTest, NewApi4) {
  mindspore::Context context;
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());
  context.MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());

  auto inner_context = std::shared_ptr<mindspore::lite::InnerContext>(mindspore::ContextUtils::Convert(&context));
  ASSERT_EQ(inner_context->device_list_.size(), 2);
  ASSERT_EQ(inner_context->device_list_.at(0).device_type_, mindspore::lite::DT_GPU);
  ASSERT_EQ(inner_context->device_list_.at(1).device_type_, mindspore::lite::DT_CPU);
}

TEST_F(MultipleDeviceTest, NewApi5) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateMultyModel1(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();

  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());

  auto model_impl = std::make_shared<mindspore::ModelImpl>();
  auto ret = model_impl->Build(content, size, mindspore::kMindIR_Lite, context);
  ASSERT_EQ(mindspore::kSuccess, ret.StatusCode());

  CheckResult(reinterpret_cast<const mindspore::lite::LiteSession *>(model_impl->GetSession())->get_kernels(),
              MultyDeviceMode1::NPU_GPU_CPU);

  /* set input data */
  std::vector<mindspore::MSTensor> inputs = model_impl->GetInputs();
  auto in = inputs[0];
  std::vector<float> in_float = {1.0};
  memcpy(in.MutableData(), in_float.data(), in.DataSize());

  std::vector<mindspore::MSTensor> outputs = model_impl->GetOutputs();

  model_impl->Predict(inputs, &outputs, nullptr, nullptr);

  /* checkout output */
  auto out = outputs[0];
  void *out_data = out.MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);

  ASSERT_LE(fabs(fp32_data[0] - (0.841471)), 0.01);
  ASSERT_LE(fabs(fp32_data[1] - (0.909297)), 0.01);
  ASSERT_LE(fabs(fp32_data[2] - (0.141120)), 0.01);
  ASSERT_LE(fabs(fp32_data[3] - (-0.756802)), 0.01);
}

TEST_F(MultipleDeviceTest, NewApi6) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateMultyModel1(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();

  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());

  auto model_impl = std::make_shared<mindspore::ModelImpl>();
  auto ret = model_impl->Build(content, size, mindspore::kMindIR_Lite, context);
  ASSERT_EQ(mindspore::kSuccess, ret.StatusCode());

  CheckResult(reinterpret_cast<const mindspore::lite::LiteSession *>(model_impl->GetSession())->get_kernels(),
              MultyDeviceMode1::CPU);

  /* set input data */
  std::vector<mindspore::MSTensor> inputs = model_impl->GetInputs();
  auto in = inputs[0];
  std::vector<float> in_float = {1.0};
  memcpy(in.MutableData(), in_float.data(), in.DataSize());

  std::vector<mindspore::MSTensor> outputs = model_impl->GetOutputs();

  model_impl->Predict(inputs, &outputs, nullptr, nullptr);

  /* checkout output */
  auto out = outputs[0];
  void *out_data = out.MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);

  ASSERT_LE(fabs(fp32_data[0] - (0.841471)), 0.01);
  ASSERT_LE(fabs(fp32_data[1] - (0.909297)), 0.01);
  ASSERT_LE(fabs(fp32_data[2] - (0.141120)), 0.01);
  ASSERT_LE(fabs(fp32_data[3] - (-0.756802)), 0.01);
}

TEST_F(MultipleDeviceTest, NewApi7) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateMultyModel2(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();

  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());

  auto model_impl = std::make_shared<mindspore::ModelImpl>();
  auto ret = model_impl->Build(content, size, mindspore::kMindIR_Lite, context);
  ASSERT_EQ(mindspore::kSuccess, ret.StatusCode());

  CheckResult(reinterpret_cast<const mindspore::lite::LiteSession *>(model_impl->GetSession())->get_kernels(),
              MultyDeviceMode1::NPU2);
}

TEST_F(MultipleDeviceTest, NewApi8) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateMultyModel2(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();

  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  // create a context
  auto context = std::make_shared<mindspore::Context>();
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());

  auto model_impl = std::make_shared<mindspore::ModelImpl>();
  auto ret = model_impl->Build(content, size, mindspore::kMindIR_Lite, context);
  ASSERT_EQ(mindspore::kSuccess, ret.StatusCode());

  CheckResult(reinterpret_cast<const mindspore::lite::LiteSession *>(model_impl->GetSession())->get_kernels(),
              MultyDeviceMode1::GPU_NPU2);
}
