/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include <tuple>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/dev.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::transform::OptionMap;

std::string ShapesToString(const ShapeArray &shapes) {
  std::stringstream buffer;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      buffer << ",";
    }
    buffer << "[";
    const auto &shape = shapes[i];
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j != 0) {
        buffer << ",";
      }
      buffer << shape[j];
    }
    buffer << "]";
  }
  return buffer.str();
}

bool IsGeTrain() {
  auto env_ge = common::GetEnv("MS_ENABLE_GE");
  auto env_training = common::GetEnv("MS_GE_TRAIN");
  if (env_ge == "1" && env_training == "1") {
    return true;
  }
  return false;
}

std::string GetGraphName(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  std::string name;
  if (kg == nullptr) {
    name = graph->ToString();
  } else {
    FuncGraphPtr origin_graph = kg->GetFuncGraph();
    MS_EXCEPTION_IF_NULL(origin_graph);
    name = origin_graph->ToString();
  }
  return name;
}

OptionMap GetComputeGraphOptions(const ShapeArray &input_shapes, bool is_dynamic_shape) {
  OptionMap options{};
  if (common::GetEnv("GE_TRAIN") == "1") {
    (void)options.emplace("ge.exec.variable_acc", "1");
  }
  if (!is_dynamic_shape) {
    return options;
  }
  (void)options.emplace("ge.exec.dynamicGraphExecuteMode", "dynamic_execute");
  (void)options.emplace("ge.exec.dataInputsShapeRange", ShapesToString(input_shapes));
  return options;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = transform::NewConverter(anf_graph);
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    transform::SetTraining(converter, false);
  }
  transform::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  transform::GenerateBroadcastGraph(converter, init_inputs_map);
  transform::GenerateCheckpointGraph(converter);
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }

  auto graph_name = GetGraphName(anf_graph);
  std::string init_graph = "init_subgraph." + graph_name;
  std::string checkpoint_name = "save." + graph_name;
  const auto options = GetComputeGraphOptions(converter->input_shapes(), converter->dynamic_shape_inputs());
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)transform::AddGraph(graph_name, transform::GetComputeGraph(converter), options);
  (void)transform::AddGraph(init_graph, transform::GetInitGraph(converter));
  (void)transform::AddGraph(BROADCAST_GRAPH_NAME, transform::GetBroadcastGraph(converter));

  transform::Status ret = transform::AddGraph(checkpoint_name, transform::GetSaveCheckpointGraph(converter));
  if (ret == transform::Status::SUCCESS) {
    transform::SetAnfGraph(checkpoint_name, anf_graph);
  }

  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
