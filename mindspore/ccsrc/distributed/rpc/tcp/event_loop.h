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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_EV_LOOP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_EV_LOOP_H_

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <semaphore.h>
#include <functional>
#include <list>
#include <mutex>
#include <queue>
#include <map>
#include <string>

namespace mindspore {
namespace distributed {
namespace rpc {
class EventLoop;
using Duration = uint64_t;

// Max epoll set size
constexpr auto EPOLL_SIZE = 4096;

// Max epoll event size
constexpr auto EPOLL_EVENTS_SIZE = 64;

typedef void (*EventHandler)(int fd, uint32_t events, void *data);
int EventLoopRun(EventLoop *evloop, int timeout);

/*
 * The event occurred on the fd.
 */
typedef struct {
  int fd;
  void *data;
  EventHandler handler;
} Event;

/*
 * The class EventLoop monitors a certain file descriptor created by eventfd function call,
 * and triggers tasks when any event occurred on the file descriptor.
 */
class EventLoop {
 public:
  EventLoop() : epoll_fd_(-1), is_stop_(false), loop_thread_(0), task_queue_event_fd_(-1) {}
  EventLoop(const EventLoop &) = delete;
  EventLoop &operator=(const EventLoop &) = delete;
  ~EventLoop() = default;

  bool Initialize(const std::string &threadName);
  void Finalize();

  // Add task (eg. send message, reconnect etc.) to task queue of the event loop.
  // These tasks are executed asynchronously.
  size_t AddTask(std::function<int()> &&task);

  // The number of tasks in the pending task queue.
  size_t RemainingTaskNum();

  // Set event handler for events(read/write/..) occurred on the socket fd.
  int SetEventHandler(int sock_fd, uint32_t events, EventHandler handler, void *data);

  // Modify the evnets monitored by epoll.
  int UpdateEpollEvent(int fd, uint32_t events);
  int DeleteEpollEvent(int fd);

 private:
  void AddEvent(Event *event);

  // Allocate the resources of epoll and tasks.
  int InitResource();

  // Release the resources of epoll and tasks.
  void ReleaseResource();

  // Stop the event loop.
  void Stop();

  // Operate the soft deleted events.
  void AddDeletedEvent(Event *event);
  int FindDeletedEvent(const Event *event);
  void RemoveDeletedEvents();

  // Event operations.
  void HandleEvent(struct epoll_event *events, size_t nevent);
  void DeleteEvent(int fd);
  Event *FindEvent(int fd);

  // Listen on some sockets.
  int epoll_fd_;

  // Whether the event loop should stop.
  bool is_stop_;

  sem_t sem_id_;
  std::mutex task_queue_mutex_;

  // The loop thread.
  pthread_t loop_thread_;

  // eventfd to trigger task_queue__.
  int task_queue_event_fd_;

  // Queue tasks like send message, reconnect, collect metrics, etc.
  // This tasks will be triggered by task_queue_event_fd_.
  std::queue<std::function<void()>> task_queue_;

  // Events on the socket.
  std::mutex event_lock_;
  std::map<int, Event *> events_;

  // To be safe, use a list to preserve deleted events rather than a map. Because the caller may
  // delete events on the same fd twice in once epoll_wait.
  std::map<int, std::list<Event *>> deleted_events_;

  friend int EventLoopRun(EventLoop *evloop, int timeout);
  friend void QueueReadyCallback(int fd, uint32_t events, void *arg);
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
