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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <tuple>
#include <Python.h>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/graph_guard/trace.h"
#include "pipeline/jit/pi/graph_guard/guard_utils.h"

namespace mindspore {
namespace jit {
namespace graph {

typedef enum _GuardLevel {
  GDeduce = 0,
  GId,
  GType,
  GAttr,
  GEqual,
} GuardLevel;

class OptGuard {
 public:
  OptGuard();
  explicit OptGuard(const std::map<std::string, bool> &config);
  virtual ~OptGuard() = default;
  /// \brief check whether the variables guarded have been modified
  /// \param[in] frame python frame
  /// \param[in] print guard
  /// \param[out] the variables have been modified
  virtual bool Check(const PyFrameObject *frame, bool print);
  /// \brief guard the variable which has trace to retrieve
  /// \param[in] frame python frame
  /// \param[in] var to trace the path to retrieve the object
  /// \param[in] tp guard level
  /// \param[in] needSpecialize to check the content of buffer
  /// \param[in] recurseDepth to check the hierarchy element access like a.b.c by depth
  /// \param[out] whether to guard successfully
  virtual bool GuardOn(TracePtr var, GuardLevel tp = GuardLevel::GDeduce, bool needSpecialize = true,
                       int recurseDepth = 0);
  /// \brief add trace from guard, traces to replace in other guard
  /// \param[in] traces to replace in other guard
  /// \param[in] other guard with traces
  virtual void AddTraceFromGuard(const std::vector<TracePtr> &traces, std::shared_ptr<OptGuard> other);
  /// \brief return the description for the guard
  virtual std::string GetDescript();
  virtual void UpdateConfig(const std::map<std::string, bool> &config);
  virtual void Backup();
  virtual void Rollback();

 protected:
  std::vector<GuardItemPtr> guardList_;
  std::vector<GuardItemPtr> backupList_;
  std::map<std::string, bool> config_;
};
using OptGuardPtr = std::shared_ptr<OptGuard>;

extern const char kSpecializeScalar[];
extern const char kSpecializeTensor[];
extern const char kSpecializeContainer[];

}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_H
