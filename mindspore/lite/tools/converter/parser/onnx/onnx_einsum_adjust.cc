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
#include "tools/converter/parser/onnx/onnx_einsum_adjust.h"
#include <string>
#include <vector>
#include <memory>
#include "ops/reshape.h"
#include "ops/primitive_c.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr const char *DELIM_COMMA = ",";
constexpr const char *DELIM_ARROW = "->";
constexpr const char *DELIM_BLANK = " ";

int CheckCanConvertToMatmul(const std::string &first_dims, const std::string &second_dims,
                            const std::string &output_dims, bool *trans_a, bool *trans_b, bool *trans_out) {
  MS_ASSERT(cnode != nullptr);
  // dimensions other than the last two dimensions and not common dimension from the right should be the same.
  // e.g. "bdn,bdm->bnm"/"bnm,bdm->bdn"/"bhid,bhjd->bhij"/"bhid,hjd->dhij"
  auto first_subdims = first_dims.substr(0, first_dims.length() - DIMENSION_2D);
  auto second_subdims = second_dims.substr(0, second_dims.length() - DIMENSION_2D);
  auto output_subdims = output_dims.substr(0, output_dims.length() - DIMENSION_2D);
  auto min_dim = first_subdims.length() < second_subdims.length() ? first_subdims.length() : second_subdims.length();
  min_dim = min_dim < output_subdims.length() ? min_dim : output_subdims.length();
  if (first_subdims.substr(first_subdims.length() - min_dim) !=
        second_subdims.substr(second_subdims.length() - min_dim) ||
      first_subdims.substr(first_subdims.length() - min_dim) !=
        output_subdims.substr(output_subdims.length() - min_dim)) {
    MS_LOG(ERROR) << "Unsupported to convert einsum to matmul.";
    return RET_ERROR;
  }

  std::function<std::string(std::string)> get_reversed_string = [](std::string str) {
    std::reverse(str.begin(), str.end());
    return str;
  };
  std::function<bool(std::string, std::string, std::string)> matched_matmul = [](std::string dim_a, std::string dim_b,
                                                                                 std::string dim_out) -> bool {
    return dim_a.at(1) == dim_b.at(0) && dim_a.at(0) == dim_out.at(0) && dim_b.at(1) == dim_out.at(1);
  };

  auto first_dim = first_dims.substr(first_dims.length() - DIMENSION_2D);
  auto second_dim = second_dims.substr(second_dims.length() - DIMENSION_2D);
  auto output_dim = output_dims.substr(output_dims.length() - DIMENSION_2D);
  std::vector<bool> trans{false, true};
  for (size_t i = 0; i < trans.size(); i++) {
    *trans_a = trans.at(i);
    auto dim_a = *trans_a ? get_reversed_string(first_dim) : first_dim;
    for (size_t j = 0; j < trans.size(); j++) {
      *trans_b = trans.at(j);
      auto dim_b = *trans_b ? get_reversed_string(second_dim) : second_dim;
      for (size_t k = 0; k < trans.size(); k++) {
        *trans_out = trans.at(k);
        auto dim_out = *trans_out ? get_reversed_string(output_dim) : output_dim;
        if (matched_matmul(dim_a, dim_b, dim_out)) {
          return RET_OK;
        }
      }
    }
  }
  return RET_ERROR;
}
}  // namespace

bool OnnxEinsumAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameEinsum))) {
      continue;
    }
    // get the second input node whose output is the padding parameter of pad.
    auto src_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_RET(src_prim != nullptr, false);
    auto equation_value = src_prim->GetAttr("equation");
    MS_CHECK_TRUE_RET(equation_value != nullptr, false);
    auto equation = GetValue<std::string>(equation_value);
    MS_CHECK_TRUE_RET(!equation.empty(), false);
    size_t index = 0;
    while ((index = equation.find(DELIM_BLANK, index)) != std::string::npos) {
      (void)equation.erase(index, 1);
    }

    auto in_out_dims = StrSplit(equation, DELIM_ARROW);
    if (in_out_dims.size() != DIMENSION_2D) {
      MS_LOG(ERROR) << "The equation of einsum must have input and output.";
      return false;
    }
    auto inputs = StrSplit(in_out_dims.front(), DELIM_COMMA);
    if (inputs.size() != DIMENSION_2D) {
      MS_LOG(ERROR) << "Only einsum with two inputs is supported.";
      return false;
    }
    auto first_dims = inputs.front();
    auto second_dims = inputs.at(1);
    auto output_dims = in_out_dims.at(1);
    MS_CHECK_TRUE_RET(!first_dims.empty() && !second_dims.empty() && !output_dims.empty(), false);

    // check can convert to scale. e.g. "bdn,d->bdn"
    if (output_dims == first_dims && first_dims.find(second_dims) != std::string::npos) {
      auto value_node = cnode->input(0)->cast<ValueNodePtr>();
      MS_ASSERT(value_node != nullptr);
      ops::ScaleFusion scale_node;
      auto scale_prim = scale_node.GetPrim();
      MS_CHECK_TRUE_MSG(scale_prim != nullptr, RET_NULL_PTR, "dst_prim is nullptr.");
      auto axis = first_dims.find(second_dims);
      scale_node.set_axis(static_cast<int64_t>(axis));
      value_node->set_value(scale_prim);
      continue;
    }

    // convert to matmul
    bool trans_a = false;
    bool trans_b = false;
    bool trans_out = false;
    if (CheckCanConvertToMatmul(first_dims, second_dims, output_dims, &trans_a, &trans_b, &trans_out) != RET_OK) {
      MS_LOG(ERROR) << "Convert einsum to matmul failed.";
      return false;
    }
    auto value_node = cnode->input(0)->cast<ValueNodePtr>();
    MS_ASSERT(value_node != nullptr);
    ops::MatMulFusion matmul_node;
    auto scale_prim = matmul_node.GetPrim();
    MS_CHECK_TRUE_MSG(scale_prim != nullptr, RET_NULL_PTR, "dst_prim is nullptr.");
    matmul_node.set_transpose_a(trans_a);
    matmul_node.set_transpose_b(trans_b);
    value_node->set_value(scale_prim);

    if (trans_out) {
      std::vector<int> perm(output_dims.size());
      std::iota(perm.begin(), perm.end(), 0);
      std::reverse(perm.end() - DIMENSION_2D, perm.end());
      auto transpose = opt::GenTransposeNode(func_graph, cnode, perm, cnode->fullname_with_scope() + "_transpose");
      if (transpose == nullptr) {
        MS_LOG(ERROR) << "create transpose failed.";
        return false;
      }

      auto manager = Manage(func_graph, true);
      if (manager == nullptr) {
        MS_LOG(ERROR) << "manager is nullptr.";
        return false;
      }
      if (!manager->Replace(cnode, transpose)) {
        MS_LOG(ERROR) << "Replace node failed.";
        return false;
      }
    }
  }
  return true;
}
}  // namespace mindspore::lite
