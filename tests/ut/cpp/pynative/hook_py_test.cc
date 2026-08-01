/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "pynative/common.h"
#include "ir/tensor_new.h"
#include "include/utils/pynative/adapter.h"
#include "pynative/backward/hook/hook_py.h"

namespace mindspore::pynative::autograd {
class TestTensorBackwardHookRegistry : public PyCommon {};

/// Feature: Tensor backward hook registry lifecycle.
/// Description: Destroy the tensor and its backward node after registering a hook without removing the handle.
/// Expectation: The backward node destructor immediately removes the registry entry.
TEST_F(TestTensorBackwardHookRegistry, RemoveRegistryEntryWhenBackwardNodeDestroyed) {
  py::gil_scoped_acquire gil;
  const auto initial_size = RegisterHook::GetTensorBackwardHookMapSize();
  auto tensor = tensor::from_scalar(1.0F);
  auto hook = py::cpp_function([](const py::object &grad) { return grad; });
  const auto handle_id = HookAdapter::RegisterTensorBackwardHook(tensor, hook);

  ASSERT_EQ(RegisterHook::GetTensorBackwardHookMapSize(), initial_size + 1);

  tensor.reset();
  EXPECT_EQ(RegisterHook::GetTensorBackwardHookMapSize(), initial_size);
  EXPECT_NO_THROW(HookAdapter::RemoveTensorBackwardHook(handle_id));
}

/// Feature: Tensor backward hook registry lifecycle.
/// Description: Explicitly remove a hook while its backward node is still alive.
/// Expectation: Removal erases the registry entry before destroying the hook and remains idempotent.
TEST_F(TestTensorBackwardHookRegistry, ExplicitRemoveRegistryEntry) {
  py::gil_scoped_acquire gil;
  const auto initial_size = RegisterHook::GetTensorBackwardHookMapSize();
  auto hook = py::cpp_function([](const py::object &grad) { return grad; });
  auto tensor_ptr = tensor::from_scalar(1.0F);
  const auto handle_id = HookAdapter::RegisterTensorBackwardHook(tensor_ptr, hook);

  ASSERT_EQ(RegisterHook::GetTensorBackwardHookMapSize(), initial_size + 1);

  HookAdapter::RemoveTensorBackwardHook(handle_id);
  EXPECT_EQ(RegisterHook::GetTensorBackwardHookMapSize(), initial_size);
  EXPECT_NO_THROW(HookAdapter::RemoveTensorBackwardHook(handle_id));
  EXPECT_EQ(RegisterHook::GetTensorBackwardHookMapSize(), initial_size);
}
}  // namespace mindspore::pynative::autograd
