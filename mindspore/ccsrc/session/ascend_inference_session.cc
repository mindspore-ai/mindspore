/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "session/ascend_inference_session.h"
#include "operator/ops.h"
#include "ir/tensor.h"
#include "ir/tensor_py.h"
#include "ir/anf.h"
#include "ir/param_value_py.h"
#include "device/kernel_runtime.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "common/trans.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "utils/config_manager.h"
#include "utils/base_ref_extends.h"

using mindspore::tensor::TensorPy;

namespace mindspore {
namespace session {
namespace {
static TypeId GetDataType(const py::buffer_info &buf) {
  if (buf.format.size() == 1) {
    switch (buf.format.front()) {
      case 'e':
      case 'f':
      case 'd':
        switch (buf.itemsize) {
          case 2:
            return TypeId::kNumberTypeFloat16;
          case 4:
            return TypeId::kNumberTypeFloat32;
          case 8:
            return TypeId::kNumberTypeFloat64;
        }
        break;
      case 'b':
      case 'h':
      case 'i':
      case 'l':
      case 'q':
        switch (buf.itemsize) {
          case 1:
            return TypeId::kNumberTypeInt8;
          case 2:
            return TypeId::kNumberTypeInt16;
          case 4:
            return TypeId::kNumberTypeInt32;
          case 8:
            return TypeId::kNumberTypeInt64;
        }
        break;
      case 'B':
      case 'H':
      case 'I':
      case 'L':
      case 'Q':
        switch (buf.itemsize) {
          case 1:
            return TypeId::kNumberTypeUInt8;
          case 2:
            return TypeId::kNumberTypeUInt16;
          case 4:
            return TypeId::kNumberTypeUInt32;
          case 8:
            return TypeId::kNumberTypeUInt64;
        }
        break;
      case '?':
        return TypeId::kNumberTypeBool;
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
  return TypeId::kTypeUnknown;
}
}  // namespace
void AscendInferenceSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                           const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  auto input_nodes = kernel_graph->inputs();

  size_t no_weight_input = 0;
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    tensor::TensorPtr tensor = nullptr;
    if (!input_nodes[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (!AnfAlgo::IsParameterWeight(pk_node)) {
      tensor = inputs[no_weight_input++];
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
}

GraphId AscendInferenceSession::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  auto graph_id = AscendSession::CompileGraph(func_graph);
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // load weight data to device
  auto input_nodes = kernel_graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    if (!input_nodes[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (AnfAlgo::IsParameterWeight(pk_node)) {
      auto param_value = std::dynamic_pointer_cast<ParamValuePy>(pk_node->default_param());
      MS_EXCEPTION_IF_NULL(param_value);
      auto py_param = param_value->value();
      MS_EXCEPTION_IF_NULL(py_param);
      py::array py_array = py_param.cast<py::array>();
      py::buffer_info buf = py_array.request();
      auto buf_type = GetDataType(buf);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(buf.size * buf.itemsize), buf_type, buf.ptr)) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
  return graph_id;
}
}  // namespace session
}  // namespace mindspore
