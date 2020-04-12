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

#include "optimizer/irpass.h"

#include <string>

#include "optimizer/irpass/symbol_resolver.h"
#include "optimizer/irpass/arithmetic_simplify.h"
#include "optimizer/irpass/special_op_eliminate.h"
#include "optimizer/irpass/item_tuple_eliminate.h"
#include "optimizer/irpass/env_item_eliminate.h"
#include "optimizer/irpass/tile_eliminate.h"
#include "optimizer/irpass/cast_eliminate.h"
#include "optimizer/irpass/reshape_eliminate.h"
#include "optimizer/irpass/transpose_eliminate.h"
#include "optimizer/irpass/reduce_eliminate.h"
#include "optimizer/irpass/partial_eliminate.h"
#include "optimizer/irpass/ref_eliminate.h"
#include "optimizer/irpass/merge_addn.h"
#include "optimizer/irpass/branch_culling.h"
#include "optimizer/irpass/gradient_eliminate.h"
#include "optimizer/irpass/minmax_grad.h"
#include "optimizer/irpass/inline.h"
#include "optimizer/irpass/convert.h"
#include "optimizer/irpass/specialize_transform.h"
#include "optimizer/irpass/incorporate_getitem.h"
#include "optimizer/irpass/incorporate_call.h"
#include "optimizer/irpass/grad_var_prepare.h"
#include "optimizer/irpass/param_replace.h"

namespace mindspore {
namespace opt {
namespace irpass {
OptimizeIRPassLib::OptimizeIRPassLib() {
  arithmetic_simplify_ = MakeSubstitution(ArithmeticSimplify(), "arithmetic_simplify",
                                          {prim::kPrimScalarAdd, prim::kPrimScalarMul, prim::kPrimTensorAdd,
                                           prim::kPrimAddN, prim::kPrimIdentity, prim::kPrimMomentum, prim::kPrimMul});
  special_op_eliminate_ = MakeSubstitution(SpecialOpEliminater(), "special_op_eliminate",
                                           {prim::kPrimInsertGradientOf, prim::kPrimPrintShapeType,
                                            prim::kPrimGetRefKey, prim::kPrimMirror, prim::kPrimVirtualDiv});
  zero_like_fill_zero_ = MakeSubstitution(ZeroLikeFillZero(), "zero_like_fill_zero", prim::kPrimZerosLikeTensor);

  // ops eliminate
  item_tuple_eliminate_ =
    MakeSubstitution(ItemTupleEliminater(), "item_tuple_eliminate", {prim::kPrimTupleGetItem, prim::kPrimTupleSetItem});
  tile_eliminate_ = MakeSubstitution(TileMultiplyByOne(), "tile_eliminate", prim::kPrimTile);
  cast_eliminate_ = MakeSubstitution(CastEliminater(), "cast_eliminate", prim::kPrimCast);
  reshape_eliminate_ = MakeSubstitution(ReshapeEliminater(), "reshape_eliminate", prim::kPrimReshape);
  transpose_eliminate_ = MakeSubstitution(TransposeSameIOEliminater(), "transpose_eliminate", prim::kPrimTranspose);
  reduce_eliminate_ = MakeSubstitution(
    ReduceOneEliminater(), "reduce_eliminate",
    {prim::kPrimReduceMean, prim::kPrimReduceAll, prim::kPrimReduceSum, prim::kPrimReduceMax, prim::kPrimReduceMin});
  partial_eliminate_ = MakeSubstitution(PartialEliminater(), "partial_eliminate", IsCNodeDup);
  same_eliminate_ = MakeSubstitution(SameEliminater(), "same_eliminate", prim::kPrimSameTypeShape);
  reset_defer_inline_ = MakeSubstitution(ResetDeferInline(), "reset_defer_inline", IsValueNode<FuncGraph>);

  // Env Item Eliminate
  new_env_get_item_ = MakeSubstitution(NewEnvGetItem(), "new_env_get_item", prim::kPrimEnvGetItem);
  add_env_get_item_ = MakeSubstitution(AddEnvGetItem(), "add_env_get_item", prim::kPrimEnvGetItem);
  env_get_set_item_ = MakeSubstitution(EnvGetSetItem(), "env_get_set_item", prim::kPrimEnvGetItem);
  incorporate_env_getitem_ =
    MakeSubstitution(IncorporateEnvGetitem(), "incorporate_env_get_item", prim::kPrimEnvGetItem);
  incorporate_env_getitem_switch_ =
    MakeSubstitution(IncorporateEnvGetitemSwitch(), "incorporate_env_getitem_switch", prim::kPrimEnvGetItem);

  // Ref eliminate
  make_ref_eliminate_ = MakeSubstitution(MakeRefEliminater(), "make_ref_eliminate", prim::kPrimMakeRef);
  get_make_ref_eliminate_ =
    MakeSubstitution(GetMakeRefEliminater(), "get_make_ref_eliminate", {prim::kPrimGetRefKey, prim::kPrimGetRefValue});
  replace_refkey_by_param_ = MakeSubstitution(ReplaceRefkeyByParam(), "replace_refkey_by_param", IsValueNode<RefKey>);
  replace_old_param_ = MakeSubstitution(ReplaceOldParam(), "replace_old_param", IsParam);

  // Gradient transforms
  expand_jprim_ = MakeSubstitution(ExpandJPrim(), "expand_jprim", prim::kPrimJ);
  stop_gradient_eliminate_ =
    MakeSubstitution(StopGradientEliminater(), "stop_gradient_eliminate", prim::kPrimStopGradient);
  minmaximum_grad_ = MakeSubstitution(MinMaximumGrad(), "minmaximum_grad", prim::kPrimTupleGetItem);

  // branch culling
  switch_simplify_ = MakeSubstitution(SwitchSimplify(), "switch_simplify", prim::kPrimSwitch);
  float_tuple_getitem_switch_ =
    MakeSubstitution(FloatTupleGetItemSwitch(), "float_tuple_getitem_switch", prim::kPrimTupleGetItem);
  float_env_getitem_switch_ =
    MakeSubstitution(FloatEnvGetItemSwitch(), "float_env_getitem_switch", prim::kPrimEnvGetItem);
  convert_switch_replacement_ = MakeSubstitution(ConvertSwitchReplacement(), "convert_switch_replacement", IsCNodeDup);

  // Addn
  merge_addn_ = MakeSubstitution(MergeAddN(), "merge_addn", prim::kPrimAddN);
  addn_zero_filter_ = MakeSubstitution(AddNZeroFilter(), "addn_zero_filter", prim::kPrimAddN);

  // inline
  inline_ = MakeSubstitution(Inliner(), "inline", IsCNodeGraph);
  replace_applicator_ = MakeSubstitution(ReplaceApplicator(), "replace_applicator", IsValueNode<FuncGraph>);
  specialize_transform_ = MakeSubstitution(SpecializeOnGraphArguments(), "specialize_transform", IsCNodeGraph);

  // Incorporation
  incorporate_getitem_ = MakeSubstitution(IncorporateGetitem(), "incorporate_getitem", prim::kPrimTupleGetItem);
  incorporate_getitem_switch_ =
    MakeSubstitution(IncorporateGetitemSwitch(), "incorporate_getitem_switch", prim::kPrimTupleGetItem);
  incorporate_call_ = MakeSubstitution(IncorporateCall(), "incorporate_call", IsCNodeDup);
  incorporate_call_switch_ = MakeSubstitution(IncorporateCallSwitch(), "incorporate_call_switch", IsCNodeDup);

  // Virtual Dataset
  virtual_dataset_eliminate_ =
    MakeSubstitution(VirtualDatasetEliminater(), "virtual_dataset_eliminate", prim::kPrimVirtualDataset);

  // Convert
  print_tuple_wrapper_ = MakeSubstitution(PrintTupleWrapper(), "print_tuple_wrapper", prim::kPrimPrint);
}

ResolveIRPassLib::ResolveIRPassLib() {
  resolver_resolve_ = MakeSubstitution(ResolverResolve(), "resolver_resolve", prim::kPrimResolve);
  resolver_getattr_ = MakeSubstitution(ResolverGetattr(), "resolver_getattr", prim::kPrimGetAttr);
}

InferenceOptPrepareLib::InferenceOptPrepareLib() {
  grad_var_prepare_ = MakeSubstitution(GradVarPrepare(), "grad_var_prepare", IsCNode);
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
