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

#include "src/litert/pass/runtime_ncx_pass.h"
#include <set>
#include <memory>
#ifdef ENABLE_RUNTIME_NCX_PASS
#include "src/litert/pass/runtime_optimizer.h"
#include "src/litert/pass/to_nchw_format.h"
#include "src/litert/pass/decrease_transpose_algo.h"
#include "src/litert/pass/infershape_pass.h"
#endif

namespace mindspore::lite::pass {
#ifdef ENABLE_RUNTIME_NCX_PASS
std::set<schema::PrimitiveType> ncxhwx_kernels = {schema::PrimitiveType_Conv2DFusion};

bool RuntimeNCXPassVaild(kernel::SubGraphKernel *subgraph) {
  if (subgraph->subgraph_type() == kernel::kNotSubGraph) {
    return false;
  }
  if (subgraph->subgraph_type() != kernel::kCpuFP32SubGraph && subgraph->subgraph_type() != kernel::kCpuFP16SubGraph) {
    return false;
  }
  for (const auto &kernel : subgraph->nodes()) {
    if (kernel->op_parameter() != nullptr) {
      if (kernel->op_parameter()->quant_type_ == schema::QuantType_AwareTraining ||
          kernel->op_parameter()->quant_type_ == schema::QuantType_PostTraining) {
        return false;
      }
    }
  }
  return true;
}
#endif

int RuntimeNCXPass(std::vector<kernel::KernelExec *> *subgraphs, std::vector<Tensor *> *tensors) {
#ifdef ENABLE_RUNTIME_NCX_PASS
  for (auto subgraph : *subgraphs) {
    if (subgraph->desc().arch == kernel::kDelegate) {
      continue;
    }
    auto graph = reinterpret_cast<kernel::SubGraphKernel *>(subgraph);
    if (!RuntimeNCXPassVaild(graph)) {
      continue;
    }

    RuntimeOptimizer optimize;
    Format runtime_format = subgraph->subgraph_type() == kernel::kCpuFP32SubGraph ? NC4HW4 : NC8HW8;
    optimize.AddPass(std::make_shared<ToNCHWFormat>(NHWC, runtime_format, ncxhwx_kernels));
    optimize.AddPass(std::make_shared<DecreaseTransposeAlgo>(runtime_format));
    optimize.AddPass(std::make_shared<Infershape>());
    auto ret = optimize.Run(graph, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Runtime pass failed.";
      return RET_ERROR;
    }
  }
#endif
  return RET_OK;
}
}  // namespace mindspore::lite::pass
