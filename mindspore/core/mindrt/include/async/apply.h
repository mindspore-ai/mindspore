/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_APPLY_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_APPLY_H

#include <utility>

namespace mindspore {
template <typename T, T... Ints>
struct IntegerSequenceBase {
  static constexpr std::size_t Size() noexcept { return sizeof...(Ints); }
};

namespace internal {
template <typename T, std::size_t N, std::size_t... Ints>
struct IntegerSequence : public IntegerSequence<T, N - 1, N - 1, Ints...> {};

template <typename T, std::size_t... Ints>
struct IntegerSequence<T, 0, Ints...> {
  using type = IntegerSequenceBase<T, Ints...>;
};
}  // namespace internal

template <typename T, std::size_t N>
using make_integer_sequence = typename internal::IntegerSequence<T, N>::type;

template <std::size_t... Ints>
using index_sequence = IntegerSequenceBase<std::size_t, Ints...>;

template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template <class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

template <typename Func, typename Tuple, std::size_t... Ints>
auto ApplyHelper(Func &&func, Tuple &&tuple, index_sequence<Ints...>)
  -> decltype(func(std::get<Ints>(std::forward<Tuple>(tuple))...)) {
  return func(std::get<Ints>(std::forward<Tuple>(tuple))...);
}

template <typename T, typename Func, typename Tuple, std::size_t... Ints>
auto ApplyHelper(T *ptr, Func &&func, Tuple &&tuple, index_sequence<Ints...>)
  -> decltype((ptr->*func)(std::get<Ints>(std::forward<Tuple>(tuple))...)) {
  return (ptr->*func)(std::get<Ints>(std::forward<Tuple>(tuple))...);
}

template <typename Func, typename Tuple>
auto Apply(Func &&func, Tuple &&tuple)
  -> decltype(ApplyHelper(std::forward<Func>(func), std::forward<Tuple>(tuple),
                          make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{})) {
  return ApplyHelper(std::forward<Func>(func), std::forward<Tuple>(tuple),
                     make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

template <typename T, typename Func, typename Tuple>
auto Apply(T *ptr, Func &&func, Tuple &&tuple)
  -> decltype(ApplyHelper(ptr, std::forward<Func>(func), std::forward<Tuple>(tuple),
                          make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{})) {
  return ApplyHelper(ptr, std::forward<Func>(func), std::forward<Tuple>(tuple),
                     make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}
}  // namespace mindspore
#endif
