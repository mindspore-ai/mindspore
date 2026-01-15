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

#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_MANAGER_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_MANAGER_H_

#include <memory>
#include <unordered_map>
#include <mutex>
#include "pynative/hyper_offload/hyper_offload_context.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {
// OffloadManager is a singleton that manages multiple HyperOffloadContext instances.
// It provides the main interface for Python side to interact with offload functionality.
class OffloadManager {
 public:
  // Get the singleton instance
  static OffloadManager &GetInstance();

  // Get or create an HyperOffloadContext by offload_key.
  // - If not ready (warm-up phase): create a new context and add to contexts_[offload_key]
  // - If ready (replay phase): return contexts_[offload_key]
  HyperOffloadContextPtr GetOrCreateOffloadCtx(size_t offload_key);

  // Called by user at the end of a single step to signal warm-up completion.
  // This will set offload_ready_ to true for all contexts.
  void Analyse();

  // Check if offload is ready (warm-up completed)
  static bool IsOffloadReady() { return offload_ready_; }

  // Reset all contexts and state
  void Reset();

  // Clear resources
  void ClearResources();

  // Get all contexts (for debugging/inspection)
  const std::unordered_map<size_t, HyperOffloadContextPtr> &contexts() const { return contexts_; }

  // Get current context count
  size_t context_count() const { return contexts_.size(); }

  // Set current active offload context (called by StartOffload)
  static void SetCurrentContext(HyperOffloadContext *ctx) { current_offload_context_ = ctx; }

  // Clear current active offload context (called by EndOffload)
  static void ClearCurrentContext() { current_offload_context_ = nullptr; }

  // Get current active offload context
  static HyperOffloadContext *GetCurrentContext() { return current_offload_context_; }

  // Fast check if there is an active context (for performance optimization)
  static bool HasActiveContext() { return current_offload_context_ != nullptr; }

 private:
  OffloadManager() = default;
  ~OffloadManager() = default;
  DISABLE_COPY_AND_ASSIGN(OffloadManager);

  // Managed HyperOffloadContext instances, keyed by offload_key
  std::unordered_map<size_t, HyperOffloadContextPtr> contexts_;

  // Static flag indicating whether warm-up is completed
  // Shared across all contexts
  inline static bool offload_ready_{false};

  // Current active offload context (set by StartOffload, cleared by EndOffload)
  inline static HyperOffloadContext *current_offload_context_{nullptr};

  // Mutex for thread safety
  std::mutex mutex_;
};

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_MANAGER_H_
