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
#include "frontend/expander/bprop/bprop_meta_func_graph.h"

#include <unordered_set>
#include "include/common/utils/utils.h"
#include "frontend/expander/bprop/bprop.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace expander {
namespace bprop {
FuncGraphPtr NewGraph(const AbstractBasePtrList &abs_list) {
  auto fg = std::make_shared<FuncGraph>();
  for (const auto &abs : abs_list) {
    auto para = fg->add_parameter();
    para->set_abstract(abs);
  }
  return fg;
}

FuncGraphPtr BpropMetaFuncGraph::GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  try {
    if (!expander::bprop::ExpandBpropInGraphMode(handle_, primal_, fg)) {
      return nullptr;
    }
  } catch (const py::type_error &ex) {
    MS_EXCEPTION(TypeError) << "Bprop \"" << primal_->name() << "\" encounter a problem: [" << ex.what() << "]";
  } catch (const py::value_error &ex) {
    MS_EXCEPTION(ValueError) << "Bprop \"" << primal_->name() << "\" encounter a problem: [" << ex.what() << "]";
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Bprop \"" << primal_->name() << "\" encounter a problem: [" << e.what() << "]";
  }
  return fg;
}

static const std::unordered_set<std::string> g_blacklist = {"SparseGatherV2",
                                                            "EmbeddingLookup",
                                                            "ExtractVolumePatches",
                                                            "AffineGrid",
                                                            "ScatterAddWithAxis",
                                                            "Expand",
                                                            "AllReduce",
                                                            "AllGather",
                                                            "_MirrorOperator",
                                                            "CropAndResize",
                                                            "RGBToHSV",
                                                            "Load",
                                                            "UpdateState",
                                                            "Depend",
                                                            "ConvertToDynamic",
                                                            "DynamicResizeNearestNeighbor",
                                                            "ParallelResizeBilinear",
                                                            "DynamicBroadcastTo",
                                                            "Svd",
                                                            "MatrixSolve",
                                                            "CholeskySolve",
                                                            "CumulativeLogsumexp",
                                                            "AvgPoolV1",
                                                            "SolveTriangular",
                                                            "Eigh",
                                                            "SparseAdd",
                                                            "CSRReduceSum",
                                                            "CSRMV",
                                                            "CSRMul",
                                                            "CSRDiv",
                                                            "COOTensorGetIndices",
                                                            "COOTensorGetValues",
                                                            "COOTensorGetDenseShape",
                                                            "CSRTensorGetIndptr",
                                                            "CSRTensorGetIndices",
                                                            "CSRTensorGetValues",
                                                            "CSRTensorGetDenseShape",
                                                            "CSRSparseMatrixToDense",
                                                            "DenseToCSRSparseMatrix",
                                                            "SparseSegmentSqrtN",
                                                            "SparseSegmentSqrtNWithNumSegments",
                                                            "SparseSegmentSum",
                                                            "SparseSegmentSumWithNumSegments",
                                                            "SparseSegmentMeanWithNumSegments",
                                                            "SparseReorder",
                                                            "SparseDenseCwiseMul",
                                                            "SparseDenseCwiseDiv",
                                                            "RaggedTensorToSparse"};
bool CanExpand(const std::string &name) {
  if (common::GetEnv("MS_DEV_DISABLE_EXPANDER_BPROP") == name) {
    return false;
  }
  if (g_blacklist.count(name)) {
    return false;
  }
  return true;
}

FuncGraphPtr GetBpropMetaFuncGraph(const PrimitivePtr &primal, const CNodePtr &cnode) {
  auto prim_name = primal->name();
  const BpropHandle *handle = BpropIRBuilderFactory::Instance().GetBuilder(prim_name);
  if (!CanExpand(prim_name) || handle == nullptr) {
    return nullptr;
  }
  std::vector<AnfNodePtr> node_lists = cnode->inputs();
  auto forward_inputs_size = cnode->inputs().size() - 1;
  for (size_t i = 1; i < node_lists.size(); i++) {
    auto input_i = node_lists[i];
    if (HasAbstractMonad(input_i)) {
      --forward_inputs_size;
    }
  }
  auto fg = std::make_shared<FuncGraph>();
  auto meta_graph = std::make_shared<BpropMetaFuncGraph>(primal, handle);
  std::vector<AnfNodePtr> inputs{NewValueNode(meta_graph)};
  for (size_t i = 0; i < forward_inputs_size; ++i) {
    (void)inputs.emplace_back(fg->add_parameter());
  }
  (void)inputs.emplace_back(fg->add_parameter());
  (void)inputs.emplace_back(fg->add_parameter());
  fg->set_output(fg->NewCNode(inputs));
  fg->set_flag(mindspore::kFuncGraphFlagMetaFuncGraphBprop, true);
  if (GetPrimitiveFlag(primal, GRAPH_FLAG_SIDE_EFFECT_BACKPROP)) {
    fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  return fg;
}
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
