/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/list_operation.h"

#include <string>
#include <memory>

#include "abstract/param_validator.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
FuncGraphPtr ListAppend::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t list_append_size_expect = 2;
  abstract::CheckArgsSize("ListAppend", args_list, list_append_size_expect);

  AbstractBasePtr obj_arg = args_list[0];
  abstract::AbstractListPtr arg0_list = dyn_cast<abstract::AbstractList>(obj_arg);
  MS_EXCEPTION_IF_NULL(arg0_list);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("append");
  AnfNodePtr arg0_node = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  size_t arg0_length = arg0_list->size();
  for (size_t i = 0; i < arg0_length; ++i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(SizeToLong(i))}));
  }
  AnfNodePtr arg1_node = ret->add_parameter();
  elems.push_back(arg1_node);

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

FuncGraphPtr ListInsert::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  const size_t list_insert_args_size = 3;
  abstract::CheckArgsSize("ListInsert", args_list, list_insert_args_size);
  AbstractBasePtr index_arg = args_list[0];
  AbstractBasePtr obj_arg = args_list[1];

  abstract::AbstractListPtr arg0_list = dyn_cast<abstract::AbstractList>(index_arg);
  MS_EXCEPTION_IF_NULL(arg0_list);
  size_t list_len = arg0_list->size();
  int64_t len = SizeToLong(list_len);
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("insert");
  AnfNodePtr arg0_node = ret->add_parameter();
  (void)ret->add_parameter();
  AnfNodePtr insert_obj_node = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  auto obj_arg_value = obj_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(obj_arg_value);
  if (!utils::isa<int64_t>(obj_arg_value)) {
    MS_EXCEPTION(TypeError) << "Integer argument expected, but got " << obj_arg_value->type_name()
                            << " type value: " << obj_arg_value->ToString();
  }
  int64_t index_value = GetValue<int64_t>(obj_arg_value);
  int64_t insert_position = 0;
  if (index_value >= len) {
    insert_position = len;
  } else if (index_value > 0 && index_value < len) {
    insert_position = index_value;
  } else if (index_value < 0 && index_value > -len) {
    insert_position = len + index_value;
  }
  for (int64_t i = 0; i < insert_position; ++i) {
    auto value = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(i)});
    elems.push_back(value);
  }
  elems.push_back(insert_obj_node);
  for (int64_t i = insert_position; i < len; ++i) {
    auto value = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(i)});
    elems.push_back(value);
  }
  auto out = ret->NewCNode(elems);
  ret->set_output(out);
  return ret;
}

FuncGraphPtr ListPop::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t list_pop_args_size = 2;
  abstract::CheckArgsSize("ListPop", args_list, list_pop_args_size);
  abstract::AbstractListPtr list_input = dyn_cast<abstract::AbstractList>(args_list[0]);
  AbstractBasePtr pop_index = args_list[1];
  MS_EXCEPTION_IF_NULL(list_input);
  size_t list_len = list_input->size();
  int64_t len = SizeToLong(list_len);
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("pop");
  AnfNodePtr arg0_node = ret->add_parameter();
  (void)ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  auto pop_index_value = pop_index->BuildValue();
  if (!utils::isa<int64_t>(pop_index_value)) {
    MS_EXCEPTION(TypeError) << "Integer argument expected, but got " << pop_index_value->type_name()
                            << " type value: " << pop_index_value->ToString();
  }
  int64_t index_value = GetValue<int64_t>(pop_index_value);
  if (index_value >= len || index_value < -1 * len) {
    MS_EXCEPTION(IndexError) << "The pop index out of range.";
  }
  int64_t pop_position = (index_value >= 0) ? index_value : (len + index_value);

  for (int64_t i = 0; i < pop_position; ++i) {
    auto value = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(i)});
    elems.push_back(value);
  }
  auto pop_node = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(pop_position)});
  for (int64_t i = pop_position + 1; i < len; ++i) {
    auto value = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg0_node, NewValueNode(i)});
    elems.push_back(value);
  }

  auto new_list = ret->NewCNode(elems);
  auto out = ret->NewCNode({NewValueNode(prim::kPrimMakeTuple), new_list, pop_node});
  ret->set_output(out);
  return ret;
}

FuncGraphPtr ListClear::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  abstract::CheckArgsSize("ListClear", args_list, 1);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("clear");
  (void)ret->add_parameter();

  auto empty_list = std::vector<ValuePtr>();
  ret->set_output(NewValueNode(std::make_shared<ValueList>(empty_list)));
  return ret;
}

FuncGraphPtr ListExtend::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  abstract::CheckArgsSize("ListExtend", args_list, 2);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("extend");

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  AddNodeToElems(args_list[0], ret, &elems);
  AddNodeToElems(args_list[1], ret, &elems);

  auto out = ret->NewCNode(elems);
  ret->set_output(out);
  return ret;
}

void ListExtend::AddNodeToElems(const AbstractBasePtr &arg, const FuncGraphPtr &ret, std::vector<AnfNodePtr> *elems) {
  abstract::AbstractListPtr arg_list = dyn_cast<abstract::AbstractList>(arg);
  MS_EXCEPTION_IF_NULL(arg_list);
  int64_t len = SizeToLong(arg_list->size());
  AnfNodePtr arg_node = ret->add_parameter();
  for (int64_t i = 0; i < len; ++i) {
    auto value = ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg_node, NewValueNode(i)});
    elems->push_back(value);
  }
}

FuncGraphPtr ListReverse::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  abstract::CheckArgsSize("ListReverse", args_list, 1);
  abstract::AbstractListPtr arg_list = dyn_cast<abstract::AbstractList>(args_list[0]);
  MS_EXCEPTION_IF_NULL(arg_list);
  int64_t arg_length = SizeToLong(arg_list->size());

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("reverse");
  AnfNodePtr arg_node = ret->add_parameter();

  std::vector<AnfNodePtr> elems;
  elems.push_back(NewValueNode(prim::kPrimMakeList));
  for (int64_t i = arg_length - 1; i >= 0; --i) {
    elems.push_back(ret->NewCNode({NewValueNode(prim::kPrimListGetItem), arg_node, NewValueNode(i)}));
  }

  ret->set_output(ret->NewCNode(elems));
  return ret;
}

FuncGraphPtr ListCount::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  const size_t list_count_args_size = 2;
  abstract::CheckArgsSize("ListCount", args_list, list_count_args_size);
  auto &list_input = args_list[0];
  auto &element_value = args_list[1];

  auto arg_list = dyn_cast_ptr<abstract::AbstractList>(list_input);
  MS_EXCEPTION_IF_NULL(arg_list);
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("count");
  (void)ret->add_parameter();
  (void)ret->add_parameter();

  ValuePtr count_value = element_value->BuildValue();
  const auto &values = arg_list->elements();
  int64_t count = 0;
  for (auto value : values) {
    if (ComparesTwoValues(count_value, value->BuildValue())) {
      ++count;
    }
  }

  auto out = NewValueNode(MakeValue(count));
  ret->set_output(out);
  return ret;
}

bool ListCount::ComparesTwoValues(const ValuePtr &count_value, const ValuePtr &list_value) {
  MS_EXCEPTION_IF_NULL(count_value);
  MS_EXCEPTION_IF_NULL(list_value);

  if (count_value->isa<AnyValue>()) {
    MS_EXCEPTION(TypeError) << "The list count not support variable scene now. The count data is Tensor type.";
  }
  if (list_value->isa<AnyValue>()) {
    MS_EXCEPTION(TypeError) << "The list count not support variable scene now. Tensor type data exists in the list.";
  }

  if (!count_value->IsSameTypeId(list_value->tid())) {
    return false;
  }
  if (count_value->isa<tensor::Tensor>()) {
    auto list_tensor_value = list_value->cast_ptr<tensor::Tensor>();
    MS_EXCEPTION_IF_NULL(list_tensor_value);
    return count_value->cast_ptr<tensor::Tensor>()->ValueEqual(*list_tensor_value);
  } else {
    return *count_value == *list_value;
  }
}
}  // namespace prim
}  // namespace mindspore
