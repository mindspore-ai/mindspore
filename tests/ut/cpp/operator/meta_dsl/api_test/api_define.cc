/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tests/ut/cpp/operator/meta_dsl/api_test/api_define.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"

namespace mindspore::prim {
/** Python code:
 *  def check_isinstance(x):
 *    return isinstance(x, Tensor), isinstance(x, (int, float))
 */
BeginFunction(TestIsInstance, x) {
  auto res1 = IsInstance(x, TypeId::kObjectTypeTensorType);
  auto res2 = IsInstance(x, {TypeId::kNumberTypeInt, TypeId::kNumberTypeFloat});
  Return(Tuple(res1, res2));
}
EndFunction(TestIsInstance)

/** Python code:
 *  def if_simple(x):
 *    if x > 0:
 *      return 1
 *    return 0
 */
BeginFunction(TestIf, x) {
  auto true_branch = [&]() { Return(Value(1)); };
  auto false_branch = [&]() { Return(Value(0)); };
  Return(If(Call(Prim(Greater), x, Value(0)), true_branch, false_branch));
}
EndFunction(TestIf)

/** Python code:
 *  def if_exp(x, y):
 *    if x is None:
 *      return x
 *    elif x < 0:
 *      return x - y
 *    return x + y
 */
BeginFunction(TestIfExp, x, y) {
  auto cond1 = IsNone(x);
  auto cond2 = Call(Prim(Less), x, Value(0));
  auto branch1 = [&]() { Return(x); };
  auto branch2 = [&]() { Return(Call(Prim(Sub), x, y)); };
  auto branch3 = [&]() { Return(Call(Prim(Add), x, y)); };
  Return(If({{cond1, branch1}, {cond2, branch2}}, branch3));
}
EndFunction(TestIfExp)

/** Python code:
 *  result = []
 *  sequence = (x, x, x, x, x)
 *  for index, item in enumerate(sequence):
 *    result.append(item + index)
 */
BeginFunction(TestFor, x) {
  auto result = List();
  auto sequence = Tuple(x, x, x, x, x);
  auto loop_func = [&](const NodePtr &index, const NodePtr &item, const NodePtr &result) {
    auto arg = Call(Prim(Add), item, index);
    auto out = Call(Prim(ListAppend), result, arg);
    Return(out);
  };
  auto out = For(loop_func, sequence, result, Value(1), Value(3));
  Return(out);
}
EndFunction(TestFor)

/** Python code:
 *  def for_func(x, lower, upper):
 *    def cumsum(index, res):
 *      return index + res
 *
 *    for i in range(lower, upper):
 *      x = cumsum(i, x)
 *    return x
 */
BeginFunction(TestForiLoop, x, lower, upper) {
  auto cumsum = [&](const NodePtr &index, const NodePtr &res) { Return(Call(Prim(Add), index, res)); };
  auto out = ForiLoop(lower, upper, cumsum, x);
  Return(out);
}
EndFunction(TestForiLoop)

/** Python code:
 *  def while_func(x):
 *    while x < 100:
 *      x = x + 1
 *    return x
 */
BeginFunction(TestWhile, x) {
  auto cond_func = [&](const NodePtr &x) { Return(Less(x, Value(100))); };
  auto loop_func = [&](const NodePtr &x) { Return(Call(Prim(Add), x, Value(1))); };
  auto out = While(cond_func, loop_func, x);
  Return(out);
}
EndFunction(TestWhile)

/** Python code:
 *  def scan_func(init, xs):
 *    def cumsum(res, elem):
 *      res = res + elem
 *      return res, res
 *
 *    res = init
 *    ys = []
 *    for x in xs:
 *      res, y = cumsum(res, x)
 *      ys.append(y)
 *    return res, ys
 */
BeginFunction(TestScan, init, xs) {
  auto cumsum = [&](const NodePtr &x, const NodePtr &elem) {
    auto res = Call(Prim(Add), x, elem);
    Return(Tuple(res, res));
  };
  auto out = Scan(cumsum, init, xs);
  Return(out);
}
EndFunction(TestScan)

/** Python code:
 *  def func(x, y):
 *    return x and y
 */
BeginFunction(TestAnd, x, y) {
  auto out = And(x, y);
  Return(out);
}
EndFunction(TestAnd)

/** Python code:
 *  def func(x, y):
 *    return x or y
 */
BeginFunction(TestOr, x, y) {
  auto out = Or(x, y);
  Return(out);
}
EndFunction(TestOr)

/** Python code:
 *  def func(x, y, dtype):
 *    out = x + y
 *    return ops.cast(out, dtype)
 */
BeginFunction(TestDtype, x, y, dtype) {
  auto out = Call(Prim(Add), x, y);
  Return(Call(Prim(Cast), out, dtype));
}
EndFunction(TestDtype)

/** Python code:
 *  def func(iterable):
 *    return all(iterable), any(iterable)
 */
BeginFunction(TestAllAny, iterable) {
  Return(Tuple(All(iterable), Any(iterable)));
}
EndFunction(TestAllAny)
}  // namespace mindspore::prim
