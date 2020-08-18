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

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/arithmetic_simplify.h"
#include "frontend/optimizer/irpass/branch_culling.h"
#include "frontend/optimizer/irpass/cast_eliminate.h"
#include "frontend/optimizer/irpass/convert.h"
#include "frontend/optimizer/irpass/env_item_eliminate.h"
#include "frontend/optimizer/irpass/grad_var_prepare.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/optimizer/irpass/inline.h"
#include "frontend/optimizer/irpass/incorporate_call.h"
#include "frontend/optimizer/irpass/incorporate_getitem.h"
#include "frontend/optimizer/irpass/item_tuple_eliminate.h"
#include "frontend/optimizer/irpass/mark_interface_fusion.h"
#include "frontend/optimizer/irpass/merge_addn.h"
#include "frontend/optimizer/irpass/minmax_grad.h"
#include "frontend/optimizer/irpass/param_replace.h"
#include "frontend/optimizer/irpass/partial_eliminate.h"
#include "frontend/optimizer/irpass/reduce_eliminate.h"
#include "frontend/optimizer/irpass/ref_eliminate.h"
#include "frontend/optimizer/irpass/reshape_eliminate.h"
#include "frontend/optimizer/irpass/special_op_eliminate.h"
#include "frontend/optimizer/irpass/specialize_transform.h"
#include "frontend/optimizer/irpass/symbol_resolver.h"
#include "frontend/optimizer/irpass/tile_eliminate.h"
#include "frontend/optimizer/irpass/transpose_eliminate.h"
#include "frontend/optimizer/irpass/value_based_eliminate.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/irpass/row_tensor_eliminate.h"
#include "frontend/optimizer/irpass/sparse_tensor_eliminate.h"
#include "frontend/optimizer/irpass/switch_layer_defer_inline.h"

namespace mindspore {
namespace opt {
namespace irpass {
OptimizeIRPassLib::OptimizeIRPassLib() {
  arithmetic_simplify_ = MakeSubstitution(std::make_shared<ArithmeticSimplify>(), "arithmetic_simplify",
                                          {prim::kPrimScalarAdd, prim::kPrimScalarMul, prim::kPrimTensorAdd,
                                           prim::kPrimIdentity, prim::kPrimMomentum, prim::kPrimMul, prim::kPrimPow});
  arithmetic_simplify2_ =
    MakeSubstitution(std::make_shared<ArithmeticSimplify2>(), "arithmetic_simplify2", {prim::kPrimMul});
  special_op_eliminate_ =
    MakeSubstitution(std::make_shared<SpecialOpEliminater>(), "special_op_eliminate",
                     {prim::kPrimInsertGradientOf, prim::kPrimStopGradient, prim::kPrimHookBackward,
                      prim::kPrimPrintShapeType, prim::kPrimGetRefValue, prim::kPrimMirror, prim::kPrimVirtualDiv});
  zero_like_fill_zero_ =
    MakeSubstitution(std::make_shared<ZeroLikeFillZero>(), "zero_like_fill_zero", prim::kPrimZerosLike);
  adjust_all_reduce_mul_add_ =
    MakeSubstitution(std::make_shared<AdjustAllReduceMulAdd>(), "adjust_all_reduce_mul_add", prim::kPrimAddN);

  // ops eliminate
  item_tuple_eliminate_ = MakeSubstitution(std::make_shared<ItemTupleEliminater>(), "item_tuple_eliminate",
                                           {prim::kPrimTupleGetItem, prim::kPrimTupleSetItem, prim::kPrimListGetItem});
  tile_eliminate_ = MakeSubstitution(std::make_shared<TileMultiplyByOne>(), "tile_eliminate", prim::kPrimTile);
  cast_eliminate_ = MakeSubstitution(std::make_shared<CastEliminater>(), "cast_eliminate", prim::kPrimCast);
  reshape_eliminate_ = MakeSubstitution(std::make_shared<ReshapeEliminater>(), "reshape_eliminate", prim::kPrimReshape);
  transpose_eliminate_ =
    MakeSubstitution(std::make_shared<TransposeSameIOEliminater>(), "transpose_eliminate", prim::kPrimTranspose);
  reduce_eliminate_ = MakeSubstitution(
    std::make_shared<ReduceOneEliminater>(), "reduce_eliminate",
    {prim::kPrimReduceMean, prim::kPrimReduceAll, prim::kPrimReduceSum, prim::kPrimReduceMax, prim::kPrimReduceMin});
  partial_eliminate_ = MakeSubstitution(std::make_shared<PartialEliminater>(), "partial_eliminate", IsCNodeDup);
  same_eliminate_ = MakeSubstitution(std::make_shared<SameEliminater>(), "same_eliminate", prim::kPrimSameTypeShape);
  check_bprop_eliminate_ =
    MakeSubstitution(std::make_shared<CheckBpropEliminater>(), "check_bprop_eliminate", prim::kPrimCheckBprop);
  reset_defer_inline_ =
    MakeSubstitution(std::make_shared<ResetDeferInline>(), "reset_defer_inline", IsValueNode<FuncGraph>);
  depend_value_elim_ = MakeSubstitution(std::make_shared<DependValueElim>(), "depend_value_elim", prim::kPrimDepend);
  all_reduce_const_elim_ =
    MakeSubstitution(std::make_shared<AllReduceConstElim>(), "reduce_all_const_elim", prim::kPrimAllReduce);

  // Env Item Eliminate
  env_get_item_eliminate_ =
    MakeSubstitution(std::make_shared<EnvGetItemEliminater>(), "env_get_item_eliminate", prim::kPrimEnvGetItem);
  new_env_get_item_ = MakeSubstitution(std::make_shared<NewEnvGetItem>(), "new_env_get_item", prim::kPrimEnvGetItem);
  incorporate_env_getitem_bypass_recursive_ =
    MakeSubstitution(std::make_shared<IncorporateEnvGetitem>(true), "incorporate_env_get_item", prim::kPrimEnvGetItem);
  incorporate_env_getitem_switch_ = MakeSubstitution(std::make_shared<IncorporateEnvGetitemSwitch>(),
                                                     "incorporate_env_getitem_switch", prim::kPrimEnvGetItem);
  incorporate_env_getitem_ =
    MakeSubstitution(std::make_shared<IncorporateEnvGetitem>(), "incorporate_env_get_item", prim::kPrimEnvGetItem);

  // Ref eliminate
  make_ref_eliminate_ =
    MakeSubstitution(std::make_shared<MakeRefEliminater>(), "make_ref_eliminate", prim::kPrimMakeRef);
  get_ref_param_eliminate_ =
    MakeSubstitution(std::make_shared<GetRefParamEliminater>(), "get_ref_param_eliminate", {prim::kPrimGetRefValue});
  get_make_ref_eliminate_ = MakeSubstitution(std::make_shared<GetMakeRefEliminater>(), "get_make_ref_eliminate",
                                             {prim::kPrimGetRefKey, prim::kPrimGetRefValue});

  replace_refkey_by_param_ = MakeSubstitution(std::make_shared<ReplaceRefkeyByParam>(), "replace_refkey_by_param",
                                              IsValueNode<RefKey>, opt::FORCE_RENORM);
  replace_old_param_ = MakeSubstitution(std::make_shared<ReplaceOldParam>(), "replace_old_param", IsParam);
  // Gradient transforms
  expand_jprim_ = MakeSubstitution(std::make_shared<ExpandJPrim>(), "expand_jprim", prim::kPrimJ);
  minmaximum_grad_ = MakeSubstitution(std::make_shared<MinMaximumGrad>(), "minmaximum_grad", prim::kPrimTupleGetItem);

  // branch culling
  switch_simplify_ = MakeSubstitution(std::make_shared<SwitchSimplify>(), "switch_simplify", prim::kPrimSwitch);
  float_tuple_getitem_switch_ = MakeSubstitution(std::make_shared<FloatTupleGetItemSwitch>(),
                                                 "float_tuple_getitem_switch", prim::kPrimTupleGetItem);
  float_env_getitem_switch_ =
    MakeSubstitution(std::make_shared<FloatEnvGetItemSwitch>(), "float_env_getitem_switch", prim::kPrimEnvGetItem);
  convert_switch_replacement_ =
    MakeSubstitution(std::make_shared<ConvertSwitchReplacement>(), "convert_switch_replacement", IsCNodeDup);

  // Addn
  merge_addn_ = MakeSubstitution(std::make_shared<MergeAddN>(), "merge_addn", prim::kPrimAddN);
  addn_zero_filter_ = MakeSubstitution(std::make_shared<AddNZeroFilter>(), "addn_zero_filter", prim::kPrimAddN);

  // inline
  inline_ = MakeSubstitution(std::make_shared<Inliner>(), "inline", IsCNodeGraph);
  inline_without_move_ = MakeSubstitution(std::make_shared<DirectInliner>(false), "inline", IsCNodeGraph);
  replace_applicator_ =
    MakeSubstitution(std::make_shared<ReplaceApplicator>(), "replace_applicator", IsValueNode<FuncGraph>);
  specialize_transform_ =
    MakeSubstitution(std::make_shared<SpecializeOnGraphArguments>(), "specialize_transform", IsCNodeGraph);

  // Incorporation
  incorporate_getitem_set_ =
    MakeSubstitution(std::make_shared<IncorporateGetitemSet>(), "incorporate_getitem_set", prim::kPrimTupleGetItem);
  incorporate_getitem_from_param_ = MakeSubstitution(std::make_shared<IncorporateGetitemFromParam>(),
                                                     "incorporate_getitem_from_param", IsCNodeGraphKernel);
  incorporate_call_ = MakeSubstitution(std::make_shared<IncorporateCall>(), "incorporate_call", IsCNodeDup);
  incorporate_call_switch_ =
    MakeSubstitution(std::make_shared<IncorporateCallSwitch>(), "incorporate_call_switch", IsCNodeDup);

  // Virtual Dataset
  virtual_dataset_eliminate_ = MakeSubstitution(std::make_shared<VirtualDatasetEliminater>(),
                                                "virtual_dataset_eliminate", prim::kPrimVirtualDataset);

  // Convert
  print_tuple_wrapper_ =
    MakeSubstitution(std::make_shared<PrintTupleWrapper>(), "print_tuple_wrapper", prim::kPrimPrint);

  // Unused parameter eliminate
  unused_parameter_eliminate_ =
    MakeSubstitution(std::make_shared<UnusedParasEliminater>(), "unused_parameter_eliminate", IsCNodeGraphKernel);
  unused_output_eliminate_ =
    MakeSubstitution(std::make_shared<UnusedOutputEliminater>(), "unused_output_eliminate", IsCNodeGraphKernel);

  // AddN eliminate
  addn_eliminate_ = MakeSubstitution(std::make_shared<AddNEliminater>(), "addn_eliminate", IsCNodeGraphKernel);

  // Mark interface fusion
  mark_interface_fusion_ =
    MakeSubstitution(std::make_shared<MarkInterfaceFusion>(), "mark_interface_fusion", prim::kPrimSelect);

  // RowTensor Eliminate
  row_tensor_eliminate_ = MakeSubstitution(
    std::make_shared<RowTensorEliminater>(), "row_tensor_eliminate",
    {prim::kPrimRowTensorGetIndices, prim::kPrimRowTensorGetValues, prim::kPrimRowTensorGetDenseShape});

  // SparseTensor Eliminate
  sparse_tensor_eliminate_ = MakeSubstitution(
    std::make_shared<SparseTensorEliminater>(), "sparse_tensor_eliminate",
    {prim::kPrimSparseTensorGetIndices, prim::kPrimSparseTensorGetValues, prim::kPrimSparseTensorGetDenseShape});

  // Value_Based Eliminate
  value_based_eliminate_ = MakeSubstitution(std::make_shared<ValueBasedEliminate>(), "value_based_eliminate",
                                            {prim::kPrimSelect, prim::kPrimMinimum, prim::kPrimMaximum});

  // switch_layer defer inline
  switch_layer_defer_inline_ =
    MakeSubstitution(std::make_shared<SwitchLayerDeferInline>(), "switch_layer_defer_inline", prim::kPrimSwitchLayer);
}

ResolveIRPassLib::ResolveIRPassLib() {
  resolver_resolve_attr_ =
    MakeSubstitution(std::make_shared<ResolveAttr>(), "resolver_resolve_attr", prim::kPrimGetAttr);
  resolver_resolve_ = MakeSubstitution(std::make_shared<ResolverResolve>(), "resolver_resolve", prim::kPrimResolve);
  resolver_getattr_ = MakeSubstitution(std::make_shared<ResolverGetattr>(), "resolver_getattr", prim::kPrimGetAttr);
}

InferenceOptPrepareLib::InferenceOptPrepareLib() {
  grad_var_prepare_ = MakeSubstitution(std::make_shared<GradVarPrepare>(), "grad_var_prepare", IsCNode);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
