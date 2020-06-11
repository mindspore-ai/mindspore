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
void AscendInferenceSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                           const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  auto input_nodes = kernel_graph->inputs();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  size_t no_weight_input = 0;
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    tensor::TensorPtr tensor = nullptr;
    if (!input_nodes[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    if (AnfAlgo::IsParameterWeight(pk_node)) {
      auto param_value = std::dynamic_pointer_cast<ParamValuePy>(pk_node->default_param());
      MS_EXCEPTION_IF_NULL(param_value);
      auto py_param = param_value->value();
      MS_EXCEPTION_IF_NULL(py_param);
      py::array py_array = py_param.cast<py::array>();
      tensor = TensorPy::MakeTensor(py_array);
    } else {
      tensor = inputs[no_weight_input++];
    }
    MS_EXCEPTION_IF_NULL(tensor);
    if (AnfAlgo::OutputAddrExist(pk_node, 0)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
      bool need_sync = false;
      if (ms_context->enable_pynative_infer()) {
        if (tensor->device_address().get() == nullptr || tensor->device_address() != device_address) {
          need_sync = true;
        }
      } else {
        if (tensor->is_dirty()) {
          need_sync = true;
        } else if (tensor->device_address() != device_address) {
          (void)tensor->data_sync();
          need_sync = true;
        }
      }
      if (need_sync) {
        if (ms_context->execution_mode() == kPynativeMode || AnfAlgo::IsParameterWeight(pk_node)) {
          tensor->set_device_address(device_address);
        }
        MS_EXCEPTION_IF_NULL(device_address);
        if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                              LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                              tensor->data_c())) {
          MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
        }
      }
    }
    tensor->set_dirty(false);
  }
}
}  // namespace session
}  // namespace mindspore
