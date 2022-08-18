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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/mul_constant_pass.h"
#include <functional>
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/fusion/mul_fusion.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kMulInputSize = 3;
}  // namespace

bool MulConstantPass::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  // this pass handle this: split with split num 1
  // after this pass, such split op will be removed.
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimMulFusion)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "cnode is nullptr!";
      continue;
    }
    auto primitive_c = ops::GetOperator<mindspore::ops::MulFusion>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr!";
      continue;
    }
    if (primitive_c->get_activation_type() != NO_ACTIVATION) {
      MS_LOG(INFO) << "mul has activation " << primitive_c->get_activation_type();
      continue;
    }
    if (cnode->size() != kMulInputSize) {
      MS_LOG(WARNING) << "MulFusion input size invalid!input size: " << cnode->size();
      continue;
    }
    auto mul_input2 = cnode->input(kInputIndexTwo);
    if (mul_input2 == nullptr) {
      MS_LOG(WARNING) << "Mul input2 is nullptr!";
      continue;
    }

    if (!IsParamOrValueNodeWithData(mul_input2)) {
      MS_LOG(INFO) << "Mul input is not const node.";
      continue;
    }
    if (cnode->input(1) == nullptr) {
      MS_LOG(ERROR) << "Mul input1 is nullptr!";
      continue;
    }
    lite::DataInfo data_info;
    auto ret = lite::FetchConstData(cnode, kInputIndexTwo, converter::kFmkTypeMs, &data_info, false);
    if (ret != lite::RET_OK) {
      MS_LOG(WARNING) << "Mul fetch second-input's data failed!";
      continue;
    }
    if (data_info.data_ptr_ == nullptr) {
      MS_LOG(WARNING) << "Mul's second-input's data is nullptr!";
      continue;
    }

    auto element_num =
      std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1L, std::multiplies<int64_t>());
    if (element_num != 1) {
      MS_LOG(INFO) << "Mul input2 has more than one data.";
      continue;
    }
    if (data_info.data_type_ != kNumberTypeFloat32) {
      MS_LOG(INFO) << "Mul input2 datatype is not fp32.";
      continue;
    }
    auto data = *static_cast<float *>(data_info.data_ptr_);
    if (data == 1.0f) {
      func_graph->manager()->Replace(node, cnode->input(1));
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
