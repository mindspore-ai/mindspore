/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <unordered_map>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "utils/context/ms_context.h"
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#include "operator/ops.h"
#include "utils/utils.h"
#include "kernel/tbe/tbe_kernel_mod.h"
#include "session/kernel_graph.h"
#include "device/kernel_info.h"
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/pattern_engine.h"
#define private public
#include "pre_activate/ascend/buffer_fusion/buffer_fusion.h"

namespace mindspore {
namespace opt {
using Primitive = mindspore::Primitive;
using session::KernelGraph;
using KernelGraphPtr = std::shared_ptr<session::KernelGraph>;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
class TestHWBufferFusion : public UT::Common {
 public:
  TestHWBufferFusion() : getPyFun_("gtest_input.pre_activate.hw_opt_test", true) {}

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

static KernelGraphPtr CreateKernelGraphForBufferFusionMultipleIn(
  uint32_t after_layers, mindspore::kernel::FusionType fusiontype = mindspore::kernel::CONVLUTION) {
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;

  std::vector<int> shp = {1, 3, 3, 4};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};

  uint32_t layerscount = 1;
  CNodePtr ptr_formerlayer;
  std::string name = "";

  // Construct first node
  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  y_tensor->set_device_info(device_info);
  tensor::TensorPtr z_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  z_tensor->set_device_info(device_info);

  auto y_const = NewValueNode(y_tensor);
  auto z_const = NewValueNode(z_tensor);
  y_const->set_abstract(y_tensor->ToAbstract());
  z_const->set_abstract(z_tensor->ToAbstract());
  g->MutableInputs()->push_back(y_const);
  g->MutableInputs()->push_back(z_const);

  auto p_conv = std::make_shared<Primitive>("Conv2D");
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};

  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_conv->set_attr("input_names", input_names_v);
  p_conv->set_attr("output_names", output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_conv));
  inputs.push_back(y_const);
  inputs.push_back(z_const);
  name = "test_conv_" + std::to_string(layerscount) + "layers_graph.dot";

  auto kernelptr_first = g->NewCNode(inputs);
  kernelptr_first->set_abstract(y_tensor->ToAbstract());
  kernelptr_first->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;

  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(fusiontype);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_first.get());
  ptr_formerlayer = kernelptr_first;

  // configure fusion successor layers
  int layer_idx = 0;
  while (after_layers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    if (layer_idx == 0) {
      tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
      x_tensor->set_device_info(device_info);

      auto x_const = NewValueNode(x_tensor);
      x_const->set_abstract(x_tensor->ToAbstract());
      std::vector<std::string> input_names = {"x", "y"};
      std::vector<std::string> output_names = {"output"};
      ValuePtr input_names_v = MakeValue(input_names);
      ValuePtr output_names_v = MakeValue(output_names);
      p_relu->set_attr("input_names", input_names_v);
      p_relu->set_attr("output_names", output_names_v);

      inputs.clear();
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(ptr_formerlayer);
      inputs.push_back(x_const);
    } else {
      std::vector<std::string> input_names = {"x"};
      std::vector<std::string> output_names = {"output"};
      ValuePtr input_names_v = MakeValue(input_names);
      ValuePtr output_names_v = MakeValue(output_names);
      p_relu->set_attr("input_names", input_names_v);
      p_relu->set_attr("output_names", output_names_v);

      inputs.clear();
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(ptr_formerlayer);
    }
    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(y_tensor->ToAbstract());
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    if (layer_idx == 0) {
      builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
      builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
    } else {
      builder.SetInputsFormat({kOpFormat_NCHW});
      builder.SetInputsDeviceType({kFloat32->type_id()});
    }

    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
    layerscount++;
    layer_idx++;
  }

  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(ptr_formerlayer);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(y_tensor->ToAbstract());

  g->set_return(ret);

  draw::Draw(name, g);

  return g;
}

static KernelGraphPtr CreateKernelGraphForBufferFusionEltwiseBeforeAndAfter(
  uint32_t before_layers, uint32_t after_layers = 3,
  mindspore::kernel::FusionType fusiontype = mindspore::kernel::SEGMENT) {
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;

  std::vector<int> shp = {1, 3, 3, 4};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};

  uint32_t layerscount = 1;
  CNodePtr ptr_formerlayer;
  std::string name = "";
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);
  g->MutableInputs()->push_back(x_const);

  while (before_layers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_relu->set_attr("input_names", input_names_v);
    p_relu->set_attr("output_names", output_names_v);

    inputs.clear();
    if (layerscount == 1) {
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(x_const);
    } else {
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(ptr_formerlayer);
    }
    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(x_abstract);
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
    layerscount++;
  }

  // Construct the conv2d node
  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  y_tensor->set_device_info(device_info);
  auto y_const = NewValueNode(y_tensor);
  y_const->set_abstract(y_tensor->ToAbstract());

  if (fusiontype == kernel::FusionType::CONVLUTION) {
    auto p_conv = std::make_shared<Primitive>("Conv2D");
    std::vector<std::string> input_names = {"x", "y"};
    std::vector<std::string> output_names = {"output"};

    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_conv->set_attr("input_names", input_names_v);
    p_conv->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_conv));
    inputs.push_back(y_const);
    inputs.push_back(ptr_formerlayer);
    name = "test_conv_" + std::to_string(layerscount) + "layers_graph.dot";
  } else {
    auto p_red_seg = std::make_shared<Primitive>("ReduceOrSegment");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};

    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_red_seg->set_attr("input_names", input_names_v);
    p_red_seg->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_red_seg));
    inputs.push_back(ptr_formerlayer);
    name = "test_regOrSeg_" + std::to_string(layerscount) + "layers_graph.dot";
  }

  auto kernelptr_first = g->NewCNode(inputs);
  kernelptr_first->set_abstract(y_tensor->ToAbstract());
  kernelptr_first->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  if (fusiontype == kernel::FusionType::CONVLUTION) {
    builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  } else {
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
  }
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(fusiontype);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_first.get());
  ptr_formerlayer = kernelptr_first;

  // configure fusion successor layers
  while (after_layers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_relu->set_attr("input_names", input_names_v);
    p_relu->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_relu));
    inputs.push_back(ptr_formerlayer);

    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(y_tensor->ToAbstract());
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
    layerscount++;
  }

  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(ptr_formerlayer);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(y_tensor->ToAbstract());
  g->set_return(ret);
  draw::Draw(name, g);
  return g;
}

static KernelGraphPtr CreateKernelGraphForBufferFusionSingleIn(
  uint32_t after_layers, mindspore::kernel::FusionType fusiontype = mindspore::kernel::CONVLUTION) {
  // build the func_graph manually, eg:
  /* CreateKernelGraphForBufferFusionSingleIn(1)
   * @mindspore
   * def f(x):
   *     z=conv2d(x, y)
   *     ret=relu(z)
   *     return ret
   */
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;

  std::vector<int> shp = {1, 3, 3, 4};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};

  uint32_t layerscount = 1;
  CNodePtr ptr_formerlayer;
  std::string name = "";

  // Construct first node
  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  y_tensor->set_device_info(device_info);
  tensor::TensorPtr z_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  z_tensor->set_device_info(device_info);

  auto y_const = NewValueNode(y_tensor);
  auto z_const = NewValueNode(z_tensor);
  y_const->set_abstract(y_tensor->ToAbstract());
  z_const->set_abstract(z_tensor->ToAbstract());
  g->MutableInputs()->push_back(y_const);
  g->MutableInputs()->push_back(z_const);

  if (fusiontype == kernel::FusionType::CONVLUTION) {
    auto p_conv = std::make_shared<Primitive>("Conv2D");
    std::vector<std::string> input_names = {"x", "y"};
    std::vector<std::string> output_names = {"output"};

    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_conv->set_attr("input_names", input_names_v);
    p_conv->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_conv));
    inputs.push_back(y_const);
    inputs.push_back(z_const);
    name = "test_conv_" + std::to_string(layerscount) + "layers_graph.dot";
  } else {
    auto p_red_seg = std::make_shared<Primitive>("ReduceOrSegment");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};

    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_red_seg->set_attr("input_names", input_names_v);
    p_red_seg->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_red_seg));
    inputs.push_back(y_const);
    name = "test_regOrSeg_" + std::to_string(layerscount) + "layers_graph.dot";
  }

  auto kernelptr_first = g->NewCNode(inputs);
  kernelptr_first->set_abstract(y_tensor->ToAbstract());
  kernelptr_first->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  if (fusiontype == kernel::FusionType::CONVLUTION) {
    builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  } else {
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
  }

  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(fusiontype);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_first.get());
  ptr_formerlayer = kernelptr_first;

  // configure fusion successor layers
  while (after_layers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_relu->set_attr("input_names", input_names_v);
    p_relu->set_attr("output_names", output_names_v);

    inputs.clear();
    inputs.push_back(NewValueNode(p_relu));
    inputs.push_back(ptr_formerlayer);

    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(y_tensor->ToAbstract());
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
    layerscount++;
  }

  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(ptr_formerlayer);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(y_tensor->ToAbstract());

  g->set_return(ret);

  draw::Draw(name, g);

  return g;
}

static KernelGraphPtr CreateKernelGraphForBufferFusion(
  uint32_t targetlayers, bool conv_flag = false,
  mindspore::kernel::FusionType fusiontype = mindspore::kernel::CONVLUTION) {
  // build the func_graph manually, eg:
  /* CreateKernelGraphForBufferFusion(3)
   * @mindspore
   * def f(x):
   *     y=relu(x)
   *     z=relu(y)
   *     ret=relu(z)
   *     return ret
   */
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // x is input tensor.
  std::vector<int> shp = {1, 3, 3, 4};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);

  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};
  x_tensor->set_device_info(device_info);

  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);
  g->MutableInputs()->push_back(x_const);

  uint32_t layerscount = 1;
  CNodePtr ptr_formerlayer;
  // configure func_graph hiden layers
  while (targetlayers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_relu->set_attr("input_names", input_names_v);
    p_relu->set_attr("output_names", output_names_v);

    inputs.clear();
    if (layerscount == 1) {
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(x_const);
    } else {
      inputs.push_back(NewValueNode(p_relu));
      inputs.push_back(ptr_formerlayer);
    }
    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(x_abstract);
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
    layerscount++;
  }
  std::string name = "test_construct_" + std::to_string(layerscount) + "layers_graph.dot";
  if (conv_flag) {
    tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    y_tensor->set_device_info(device_info);
    tensor::TensorPtr z_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    z_tensor->set_device_info(device_info);
    auto y_const = NewValueNode(y_tensor);
    auto z_const = NewValueNode(y_tensor);

    y_const->set_abstract(y_tensor->ToAbstract());
    z_const->set_abstract(z_tensor->ToAbstract());

    g->MutableInputs()->push_back(y_const);

    if (fusiontype == kernel::FusionType::CONVLUTION) {
      auto p_conv = std::make_shared<Primitive>("Conv2D");
      std::vector<std::string> input_names = {"x", "y"};
      std::vector<std::string> output_names = {"output"};

      ValuePtr input_names_v = MakeValue(input_names);
      ValuePtr output_names_v = MakeValue(output_names);
      p_conv->set_attr("input_names", input_names_v);
      p_conv->set_attr("output_names", output_names_v);

      inputs.clear();
      inputs.push_back(NewValueNode(p_conv));
      inputs.push_back(y_const);
      inputs.push_back(ptr_formerlayer);
    } else {
      auto p_conv = std::make_shared<Primitive>("ReduceOrSegment");
      std::vector<std::string> input_names = {"x"};
      std::vector<std::string> output_names = {"output"};

      ValuePtr input_names_v = MakeValue(input_names);
      ValuePtr output_names_v = MakeValue(output_names);
      p_conv->set_attr("input_names", input_names_v);
      p_conv->set_attr("output_names", output_names_v);

      inputs.clear();
      inputs.push_back(NewValueNode(p_conv));
      inputs.push_back(ptr_formerlayer);
    }

    auto kernelptr_conv = g->NewCNode(inputs);
    kernelptr_conv->set_abstract(x_abstract);
    kernelptr_conv->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    if (fusiontype == kernel::FusionType::CONVLUTION) {
      builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
      builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
    } else {
      builder.SetInputsFormat({kOpFormat_NCHW});
      builder.SetInputsDeviceType({kFloat32->type_id()});
    }
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(fusiontype);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_conv.get());
    ptr_formerlayer = kernelptr_conv;
    name = "test_conv_" + std::to_string(layerscount) + "layers_graph.dot";
  }
  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(ptr_formerlayer);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(x_abstract);

  g->set_return(ret);

  draw::Draw(name, g);

  return g;
}

CNodePtr CreateKernelGraphBranch(KernelGraphPtr g, CNodePtr inputptr, int layers,
                                 const kernel::FusionType fusiontype = kernel::FusionType::CONVLUTION) {
  std::vector<int> shp = {1, 3, 3, 4};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};
  x_tensor->set_device_info(device_info);
  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);

  CNodePtr ptr_formerlayer = inputptr;
  while (layers--) {
    auto p_relu = std::make_shared<Primitive>("ReLU6");
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    ValuePtr input_names_v = MakeValue(input_names);
    ValuePtr output_names_v = MakeValue(output_names);
    p_relu->set_attr("input_names", input_names_v);
    p_relu->set_attr("output_names", output_names_v);

    std::vector<AnfNodePtr> inputs;
    inputs.clear();
    inputs.push_back(NewValueNode(p_relu));
    inputs.push_back(ptr_formerlayer);
    auto kernelptr_floor = g->NewCNode(inputs);
    kernelptr_floor->set_abstract(x_abstract);
    kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_NCHW});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetKernelType(KernelType::TBE_KERNEL);
    builder.SetFusionType(kernel::FusionType::ELEMWISE);
    builder.SetProcessor(kernel::Processor::AICORE);
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_floor.get());
    ptr_formerlayer = kernelptr_floor;
  }

  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  y_tensor->set_device_info(device_info);
  tensor::TensorPtr z_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  z_tensor->set_device_info(device_info);
  auto y_const = NewValueNode(y_tensor);
  auto z_const = NewValueNode(y_tensor);

  y_const->set_abstract(y_tensor->ToAbstract());
  z_const->set_abstract(z_tensor->ToAbstract());

  g->MutableInputs()->push_back(y_const);

  auto p_conv = std::make_shared<Primitive>("Conv2D");
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};

  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_conv->set_attr("input_names", input_names_v);
  p_conv->set_attr("output_names", output_names_v);

  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  inputs.push_back(NewValueNode(p_conv));
  inputs.push_back(y_const);
  inputs.push_back(ptr_formerlayer);

  auto kernelptr_conv = g->NewCNode(inputs);
  kernelptr_conv->set_abstract(x_abstract);
  kernelptr_conv->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(fusiontype);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_conv.get());
  return kernelptr_conv;
}

static KernelGraphPtr CreateKernelGraphForMultiUse(uint32_t targetlayer1s, uint32_t targetlayer2s) {
  /*  @mindspore
   * def f(x):
   *     multi_use=relu(x)
   *     y=relu(multi_use)
   *     z=relu(multi_use)
   *     ret=relu(y, z)
   *     return ret
   */
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // x is input tensor.
  std::vector<int> shp = {1, 3, 3, 4};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};
  x_tensor->set_device_info(device_info);

  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);

  g->MutableInputs()->push_back(x_const);

  auto p_multi = std::make_shared<Primitive>("MULTI_USE_ReLU6");
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_multi->set_attr("input_names", input_names_v);
  p_multi->set_attr("output_names", output_names_v);
  inputs.clear();
  inputs.push_back(NewValueNode(p_multi));
  inputs.push_back(x_const);
  auto kernelptr_multi = g->NewCNode(inputs);
  kernelptr_multi->set_abstract(x_abstract);
  kernelptr_multi->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_multi.get());

  CNodePtr outptrbranch1 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer2s);
  CNodePtr outptrbranch2 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer1s);

  auto p_relu = std::make_shared<Primitive>("ReLU6");
  input_names = {"x"};
  output_names = {"output"};
  input_names_v = MakeValue(input_names);
  output_names_v = MakeValue(output_names);
  p_relu->set_attr("input_names", input_names_v);
  p_relu->set_attr("output_names", output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_relu));
  inputs.push_back(outptrbranch1);
  inputs.push_back(outptrbranch2);
  auto kernelptr_floor = g->NewCNode(inputs);
  kernelptr_floor->set_abstract(x_abstract);
  kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
  builder1.SetOutputsFormat({kOpFormat_NCHW});
  builder1.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder1.SetOutputsDeviceType({kFloat32->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::ELEMWISE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), kernelptr_floor.get());

  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(kernelptr_floor);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(x_abstract);

  g->set_return(ret);
  string name = "multi_use_graph.dot";
  draw::Draw(name, g);

  return g;
}
#ifdef BUFFER_FUSION_MULTI_OUT
static KernelGraphPtr CreateKernelGraphForMultiOutputWithLinearInput(
  uint32_t targetlayer1s, uint32_t targetlayer2s, bool use_flag = true,
  const kernel::FusionType fusion_type = kernel::FusionType::CONVLUTION) {
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // x is input tensor.
  std::vector<int> shp = {1, 3, 3, 4};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};
  x_tensor->set_device_info(device_info);

  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);
  g->MutableInputs()->push_back(x_const);

  auto p_relu0 = std::make_shared<Primitive>("ReLU6");
  std::vector<std::string> input_names0 = {"x"};
  std::vector<std::string> output_names0 = {"output"};
  ValuePtr input_names_v0 = MakeValue(input_names0);
  ValuePtr output_names_v0 = MakeValue(output_names0);
  p_relu0->set_attr("input_names", input_names_v0);
  p_relu0->set_attr("output_names", output_names_v0);
  inputs.clear();
  inputs.push_back(NewValueNode(p_relu0));
  inputs.push_back(x_const);
  auto kernelptr_floor0 = g->NewCNode(inputs);
  kernelptr_floor0->set_abstract(x_abstract);
  kernelptr_floor0->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder0;
  builder0.SetInputsFormat({kOpFormat_NCHW});
  builder0.SetOutputsFormat({kOpFormat_NCHW});
  builder0.SetInputsDeviceType({kFloat32->type_id()});
  builder0.SetOutputsDeviceType({kFloat32->type_id()});
  builder0.SetKernelType(KernelType::TBE_KERNEL);
  builder0.SetFusionType(kernel::FusionType::ELEMWISE);
  builder0.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder0.Build(), kernelptr_floor0.get());
  CNodePtr ptr_formerlayer;
  ptr_formerlayer = kernelptr_floor0;

  auto p_multi = std::make_shared<Primitive>("MULTI_USE_ReLU6");
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_multi->set_attr("input_names", input_names_v);
  p_multi->set_attr("output_names", output_names_v);
  inputs.clear();
  inputs.push_back(NewValueNode(p_multi));
  inputs.push_back(ptr_formerlayer);
  auto kernelptr_multi = g->NewCNode(inputs);
  kernelptr_multi->set_abstract(x_abstract);
  kernelptr_multi->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW});
  builder.SetOutputsFormat({kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_multi.get());

  CNodePtr outptrbranch2 = nullptr;
  CNodePtr outptrbranch1 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer2s, fusion_type);
  if (use_flag) {
    outptrbranch2 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer1s, fusion_type);
  }
  auto p_relu = std::make_shared<Primitive>("ReLU6");
  input_names = {"x"};
  output_names = {"output"};
  input_names_v = MakeValue(input_names);
  output_names_v = MakeValue(output_names);
  p_relu->set_attr("input_names", input_names_v);
  p_relu->set_attr("output_names", output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_relu));
  inputs.push_back(outptrbranch1);
  if (use_flag) {
    inputs.push_back(outptrbranch2);
  }

  auto kernelptr_floor = g->NewCNode(inputs);
  kernelptr_floor->set_abstract(x_abstract);
  kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder1;
  if (use_flag) {
    builder1.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
    builder1.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  } else {
    builder1.SetInputsFormat({kOpFormat_NCHW});
    builder1.SetInputsDeviceType({kFloat32->type_id()});
  }
  builder1.SetOutputsFormat({kOpFormat_NCHW});
  builder1.SetOutputsDeviceType({kFloat32->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::ELEMWISE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), kernelptr_floor.get());
  cout << "built two branches done" << endl;
  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(kernelptr_floor);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(x_abstract);

  g->set_return(ret);
  string name = "multi_use_graph.dot";
  draw::Draw(name, g);

  return g;
}

static KernelGraphPtr CreateKernelGraphForMultiOutput(
  uint32_t targetlayer1s, uint32_t targetlayer2s, bool use_flag = true,
  const kernel::FusionType fusion_type = kernel::FusionType::CONVLUTION) {
  KernelGraphPtr g = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // x is input tensor.
  std::vector<int> shp = {1, 3, 3, 4};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NCHW, tensor_type};
  x_tensor->set_device_info(device_info);

  auto x_abstract = x_tensor->ToAbstract();
  auto x_const = NewValueNode(x_tensor);
  x_const->set_abstract(x_abstract);
  g->MutableInputs()->push_back(x_const);

  auto p_multi = std::make_shared<Primitive>("MULTI_USE_ReLU6");
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  ValuePtr input_names_v = MakeValue(input_names);
  ValuePtr output_names_v = MakeValue(output_names);
  p_multi->set_attr("input_names", input_names_v);
  p_multi->set_attr("output_names", output_names_v);
  inputs.clear();
  inputs.push_back(NewValueNode(p_multi));
  inputs.push_back(x_const);
  auto kernelptr_multi = g->NewCNode(inputs);
  kernelptr_multi->set_abstract(x_abstract);
  kernelptr_multi->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_NCHW});
  builder.SetOutputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id(), kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), kernelptr_multi.get());

  CNodePtr outptrbranch2 = nullptr;
  CNodePtr outptrbranch1 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer2s, fusion_type);
  if (use_flag) {
    outptrbranch2 = CreateKernelGraphBranch(g, kernelptr_multi, targetlayer1s, fusion_type);
  }
  auto p_relu = std::make_shared<Primitive>("ReLU6");
  input_names = {"x"};
  output_names = {"output"};
  input_names_v = MakeValue(input_names);
  output_names_v = MakeValue(output_names);
  p_relu->set_attr("input_names", input_names_v);
  p_relu->set_attr("output_names", output_names_v);

  inputs.clear();
  inputs.push_back(NewValueNode(p_relu));
  inputs.push_back(outptrbranch1);
  if (use_flag) {
    inputs.push_back(outptrbranch2);
  }
  auto kernelptr_floor = g->NewCNode(inputs);
  kernelptr_floor->set_abstract(x_abstract);
  kernelptr_floor->set_kernel_info(std::make_shared<device::KernelInfo>());
  KernelBuildInfoBuilder builder1;
  if (use_flag) {
    builder1.SetInputsFormat({kOpFormat_NCHW, kOpFormat_NCHW});
    builder1.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  } else {
    builder1.SetInputsFormat({kOpFormat_NCHW});
    builder1.SetInputsDeviceType({kFloat32->type_id()});
  }
  builder1.SetOutputsFormat({kOpFormat_NCHW});
  builder1.SetOutputsDeviceType({kFloat32->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::ELEMWISE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), kernelptr_floor.get());

  // return res
  auto p_return = std::make_shared<Primitive>("return");
  inputs.clear();
  inputs.push_back(NewValueNode(p_return));
  inputs.push_back(kernelptr_floor);
  auto ret = g->NewCNode(inputs);
  ret->set_abstract(x_abstract);

  g->set_return(ret);
  string name = "multi_use_graph.dot";
  draw::Draw(name, g);

  return g;
}
#endif
TEST_F(TestHWBufferFusion, BufferFusionlayerSingleIn1) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionSingleIn(1);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionlayerSingleIn1.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 8);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionlayerSingleIn1.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 6);
}

TEST_F(TestHWBufferFusion, BufferFusionlayerSingleIn2) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionSingleIn(2);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionlayerSingleIn2.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionlayerSingleIn2.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 6);
}

TEST_F(TestHWBufferFusion, BufferFusionlayerSingleIn3) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionSingleIn(3);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionlayerSingleIn3.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 12);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionlayerSingleIn3.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 6);
}

TEST_F(TestHWBufferFusion, BufferFusionlayer1) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(1);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
  buffer_fusion.Run(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionlayer2) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(2);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 7);
  buffer_fusion.Run(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionlayer4) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(4);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 11);
  buffer_fusion.Run(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionlayer6) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(6);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 15);
  buffer_fusion.Run(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 7);
}

TEST_F(TestHWBufferFusion, BufferFusionlayer8) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(8);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 19);
  buffer_fusion.Run(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 7);
}

TEST_F(TestHWBufferFusion, BufferFusionconv1) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(1, true);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), false);
}

TEST_F(TestHWBufferFusion, BufferFusionconv8) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(8, true);
  draw::Draw("before_BufferFusionconv8.dot", graph_ptr);

  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  std::vector<int32_t> fusion_ids;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    fusion_ids.push_back(buffer_fusion_info.first);
  }
  std::sort(fusion_ids.begin(), fusion_ids.end());
  for (auto &fusion_id : fusion_ids) {
    buffer_fusion.ReplaceFusionOp(buffer_fusion_infos[fusion_id], kernel_ptr, graph_ptr.get());
  }
  draw::Draw("after_BufferFusionconv8.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 10);
}

#ifdef BUFFER_FUSION_MULTI_OUT
TEST_F(TestHWBufferFusion, BufferFusionMultiOutWithLinearInput) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForMultiOutputWithLinearInput(1, 1, true, mindspore::kernel::OPAQUE);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 19);

  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    EXPECT_EQ(buffer_fusion_info.second.anf_nodes.size(), 3);
    EXPECT_EQ(buffer_fusion_info.second.inputs_list.size(), 1);
    EXPECT_EQ(buffer_fusion_info.second.outputs_list.size(), 2);
    buffer_fusion.ReplaceFusionOp(buffer_fusion_info.second, kernel_ptr, graph_ptr.get());
  }
  ASSERT_EQ(manager->all_nodes().size(), 21);
}

TEST_F(TestHWBufferFusion, BufferFusionMultiOut) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForMultiOutput(1, 1, true, mindspore::kernel::OPAQUE);
  draw::Draw("before_BufferFusionMultiOut.dot", graph_ptr);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 17);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    EXPECT_EQ(buffer_fusion_info.second.anf_nodes.size(), 2);
    EXPECT_EQ(buffer_fusion_info.second.inputs_list.size(), 1);
    EXPECT_EQ(buffer_fusion_info.second.outputs_list.size(), 2);
    buffer_fusion.ReplaceFusionOp(buffer_fusion_info.second, kernel_ptr, graph_ptr.get());
  }
  draw::Draw("after_BufferFusionMultiOut.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 21);
}
#endif

TEST_F(TestHWBufferFusion, BufferMultiUse) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForMultiUse(3, 4);
  draw::Draw("before_BufferMultiUse.dot", graph_ptr);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  std::vector<int32_t> fusion_ids;
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    fusion_ids.push_back(buffer_fusion_info.first);
  }
  std::sort(fusion_ids.begin(), fusion_ids.end());
  for (auto &fusion_id : fusion_ids) {
    buffer_fusion.ReplaceFusionOp(buffer_fusion_infos[fusion_id], kernel_ptr, graph_ptr.get());
  }
  draw::Draw("after_BufferMultiUse.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 15);
}

TEST_F(TestHWBufferFusion, BufferFusionReduce) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(2, true, mindspore::kernel::COMMREDUCE);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    EXPECT_EQ(buffer_fusion_info.second.anf_nodes.size(), 3);
    EXPECT_EQ(buffer_fusion_info.second.inputs_list.size(), 1);
    EXPECT_EQ(buffer_fusion_info.second.outputs_list.size(), 1);
    buffer_fusion.ReplaceFusionOp(buffer_fusion_info.second, kernel_ptr, graph_ptr.get());
  }
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionSegment) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusion(2, true, mindspore::kernel::SEGMENT);
  ASSERT_TRUE(nullptr != graph_ptr);
  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(buffer_fusion.MatchBufferFusionPattern(*graph_ptr), true);
  kernel::KernelPackPtr kernel_pack = std::make_shared<kernel::KernelPack>();
  auto kernel_ptr = std::make_shared<kernel::TbeKernelMod>(kernel_pack);
  std::unordered_map<int, BufferFusionInfo_t> buffer_fusion_infos;
  buffer_fusion.GetBufferFusionInfo(*graph_ptr, &buffer_fusion_infos);
  for (auto &buffer_fusion_info : buffer_fusion_infos) {
    EXPECT_EQ(buffer_fusion_info.second.anf_nodes.size(), 3);
    EXPECT_EQ(buffer_fusion_info.second.inputs_list.size(), 1);
    EXPECT_EQ(buffer_fusion_info.second.outputs_list.size(), 1);
    buffer_fusion.ReplaceFusionOp(buffer_fusion_info.second, kernel_ptr, graph_ptr.get());
  }
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionEltwise1BeforeAnd3After) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionEltwiseBeforeAndAfter(1);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionEltwiseBeforeAndAfter1.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 13);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionEltwiseBeforeAndAfter1.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionEltwise2BeforeAnd3After) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionEltwiseBeforeAndAfter(2);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionEltwiseBeforeAndAfter2.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 15);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionEltwiseBeforeAndAfter2.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionEltwise3BeforeAnd3After) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionEltwiseBeforeAndAfter(3);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionEltwiseBeforeAndAfter3.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 17);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionEltwiseBeforeAndAfter3.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 5);
}

TEST_F(TestHWBufferFusion, BufferFusionMultipleIn) {
  KernelGraphPtr graph_ptr = CreateKernelGraphForBufferFusionMultipleIn(2);
  ASSERT_TRUE(nullptr != graph_ptr);
  draw::Draw("before_BufferFusionMultipleIn.dot", graph_ptr);

  mindspore::opt::BufferFusion buffer_fusion = BufferFusion();
  std::vector<FuncGraphPtr> graphs{graph_ptr};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 11);
  buffer_fusion.Run(graph_ptr);
  draw::Draw("after_BufferFusionMultipleIn.dot", graph_ptr);
  ASSERT_EQ(manager->all_nodes().size(), 7);
}
}  // namespace opt
}  // namespace mindspore
