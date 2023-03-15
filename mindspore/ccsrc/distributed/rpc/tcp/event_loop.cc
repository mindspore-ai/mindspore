/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "distributed/rpc/tcp/event_loop.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <securec.h>
#include <unistd.h>
#include <utility>
#include <atomic>
#include <string>
#include <thread>

#include "actor/log.h"
#include "utils/convert_utils_base.h"
#include "include/backend/distributed/rpc/tcp/constants.h"

namespace mindspore {
namespace distributed {
namespace rpc {
int EventLoopRun(EventLoop *evloop, int timeout) {
  if (evloop == nullptr) {
    return RPC_ERROR;
  }
  struct epoll_event *events = nullptr;
  (void)sem_post(&evloop->sem_id_);

  size_t size = sizeof(struct epoll_event) * EPOLL_EVENTS_SIZE;
  events = (struct epoll_event *)malloc(size);
  if (events == nullptr) {
    MS_LOG(ERROR) << "Failed to call malloc events";
    return RPC_ERROR;
  }
  if (memset_s(events, size, 0, size) != EOK) {
    MS_LOG(ERROR) << "Failed to call memset_s.";
    free(events);
    return RPC_ERROR;
  }

  while (!evloop->is_stop_) {
    /* free deleted event handlers */
    evloop->RemoveDeletedEvents();
    int nevent = epoll_wait(evloop->epoll_fd_, events, EPOLL_EVENTS_SIZE, timeout);
    if (nevent < 0) {
      if (errno != EINTR) {
        MS_LOG(ERROR) << "Failed to call epoll_wait, epoll_fd_: " << evloop->epoll_fd_ << ", errno: " << errno;
        free(events);
        return RPC_ERROR;
      } else {
        continue;
      }
    } else if (nevent > 0) {
      /* save the epoll modify in "stop" while dispatching handlers */
      evloop->HandleEvent(events, IntToSize(nevent));
    } else {
      MS_LOG(ERROR) << "Failed to call epoll_wait, epoll_fd_: " << evloop->epoll_fd_ << ", ret: 0,errno: " << errno;
      evloop->is_stop_ = true;
    }
    if (evloop->is_stop_) {
      /* free deleted event handlers */
      evloop->RemoveDeletedEvents();
    }
  }
  evloop->is_stop_ = false;
  MS_LOG(INFO) << "Event epoll loop run end";
  free(events);

  return RPC_OK;
}

void *EvloopRun(void *arg) {
  if (arg == nullptr) {
    MS_LOG(ERROR) << "Arg is null";
  } else {
    (void)EventLoopRun(reinterpret_cast<EventLoop *>(arg), -1);
  }
  return nullptr;
}

void QueueReadyCallback(int fd, uint32_t events, void *arg) {
  EventLoop *evloop = reinterpret_cast<EventLoop *>(arg);
  if (evloop == nullptr) {
    MS_LOG(ERROR) << "The evloop is null fd:" << fd << ",events:" << events;
    return;
  }
  uint64_t count;
  ssize_t retval = read(evloop->task_queue_event_fd_, &count, sizeof(count));
  if (retval > 0 && retval == sizeof(count)) {
    // take out functions from the queue
    std::queue<std::function<void()>> q;

    evloop->task_queue_mutex_.lock();
    evloop->task_queue_.swap(q);
    evloop->task_queue_mutex_.unlock();

    // invoke functions in the queue
    while (!q.empty()) {
      q.front()();
      q.pop();
    }
  }
}

void EventLoop::ReleaseResource() {
  if (task_queue_event_fd_ != -1) {
    if (close(task_queue_event_fd_) != 0) {
      MS_LOG(ERROR) << "Failed to close task queue event fd: " << task_queue_event_fd_;
    }
    task_queue_event_fd_ = -1;
  }
  if (epoll_fd_ != -1) {
    if (close(epoll_fd_) != 0) {
      MS_LOG(ERROR) << "Failed to close epoll fd: " << epoll_fd_;
    }
    epoll_fd_ = -1;
  }
}

size_t EventLoop::AddTask(std::function<int()> &&task) {
  // put func to the queue
  task_queue_mutex_.lock();
  (void)task_queue_.emplace(std::move(task));

  // return the queue size to send's caller.
  auto result = task_queue_.size();
  task_queue_mutex_.unlock();

  if (result == 1) {
    // wakeup event loop
    uint64_t one = 1;
    ssize_t retval = write(task_queue_event_fd_, &one, sizeof(one));
    if (retval <= 0 || retval != sizeof(one)) {
      MS_LOG(WARNING) << "Failed to write queue Event fd: " << task_queue_event_fd_ << ",errno:" << errno;
    }
  }
  return result;
}

size_t EventLoop::RemainingTaskNum() {
  task_queue_mutex_.lock();
  auto task_num = task_queue_.size();
  task_queue_mutex_.unlock();
  return task_num;
}

bool EventLoop::Initialize(const std::string &threadName) {
  int retval = InitResource();
  if (retval != RPC_OK) {
    return false;
  }
  (void)sem_init(&sem_id_, 0, 0);

  if (pthread_create(&loop_thread_, nullptr, EvloopRun, reinterpret_cast<void *>(this)) != 0) {
    MS_LOG(ERROR) << "Failed to call pthread_create";
    Finalize();
    return false;
  }

  // wait EvloopRun
  (void)sem_wait(&sem_id_);
#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 12
  std::string name = threadName;

  if (name.empty()) {
    name = "EventLoopThread";
  }
  retval = pthread_setname_np(loop_thread_, name.c_str());
  if (retval != 0) {
    MS_LOG(INFO) << "Set pthread name fail name:" << name.c_str() << ",retval:" << retval;
  } else {
    MS_LOG(INFO) << "Set pthread name success name:" << name.c_str() << ",loop_thread_:" << loop_thread_;
  }
#endif

  return true;
}

void EventLoop::Finalize() {
  if (loop_thread_ > 0) {
    void *threadResult = nullptr;
    Stop();

    int ret = pthread_join(loop_thread_, &threadResult);
    if (ret != 0) {
      MS_LOG(INFO) << "Failed to call pthread_join loop_thread_";
    }
    loop_thread_ = 0;
  }

  RemoveDeletedEvents();
  ReleaseResource();
  MS_LOG(INFO) << "Stop loop succ";
}

void EventLoop::DeleteEvent(int fd) {
  auto iter = events_.find(fd);
  if (iter == events_.end()) {
    MS_LOG(INFO) << "Not found event fd:" << fd;
    return;
  }

  Event *eventData = iter->second;
  if (eventData != nullptr) {
    delete eventData;
  }
  (void)events_.erase(fd);
}

Event *EventLoop::FindEvent(int fd) {
  auto iter = events_.find(fd);
  if (iter == events_.end()) {
    return nullptr;
  }
  return iter->second;
}

int EventLoop::InitResource() {
  int retval = 0;
  is_stop_ = false;
  epoll_fd_ = epoll_create(EPOLL_SIZE);
  if (epoll_fd_ == -1) {
    MS_LOG(ERROR) << "Failed to call epoll_create, errno:" << errno;
    ReleaseResource();
    return RPC_ERROR;
  }

  // create eventfd
  task_queue_event_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (task_queue_event_fd_ == -1) {
    MS_LOG(ERROR) << "Failed to call eventfd, errno:" << errno;
    ReleaseResource();
    return RPC_ERROR;
  }

  retval = SetEventHandler(task_queue_event_fd_, EPOLLIN | EPOLLHUP | EPOLLERR, QueueReadyCallback,
                           reinterpret_cast<void *>(this));
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Add queue event fail task_queue_event_fd_:" << task_queue_event_fd_;
    ReleaseResource();
    return RPC_ERROR;
  }
  return RPC_OK;
}

int EventLoop::SetEventHandler(int fd, uint32_t events, EventHandler handler, void *data) {
  struct epoll_event ev;
  Event *evdata = nullptr;
  int ret = 0;

  if (memset_s(&ev, sizeof(ev), 0, sizeof(ev)) != EOK) {
    MS_LOG(ERROR) << "Failed to call memset_s.";
    return RPC_ERROR;
  }
  ev.events = events;

  evdata = new (std::nothrow) Event();
  if (evdata == nullptr) {
    MS_LOG(ERROR) << "Failed to call malloc eventData, fd:" << fd << ",epollfd:" << epoll_fd_;
    return RPC_ERROR;
  }

  evdata->data = data;
  evdata->handler = handler;
  evdata->fd = fd;

  event_lock_.lock();
  AddEvent(evdata);
  event_lock_.unlock();

  ev.data.ptr = evdata;
  ret = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev);
  if (ret > 0) {
    event_lock_.lock();
    DeleteEvent(fd);
    event_lock_.unlock();

    if (errno != EEXIST) {
      MS_LOG(ERROR) << "Failed to call epoll add, fail fd:" << fd << ",epollfd:" << epoll_fd_ << ",errno:" << errno;
    } else {
      MS_LOG(ERROR) << "The fd already existed in epoll, fd:" << fd << ",epollfd:" << epoll_fd_ << ",errno:" << errno;
    }
    return RPC_ERROR;
  }
  return RPC_OK;
}

void EventLoop::AddEvent(Event *event) {
  if (event == nullptr) {
    return;
  }
  DeleteEvent(event->fd);
  (void)events_.emplace(event->fd, event);
}

int EventLoop::DeleteEpollEvent(int fd) {
  Event *tev = nullptr;
  struct epoll_event ev;
  int ret = 0;

  event_lock_.lock();
  tev = FindEvent(fd);
  if (tev == nullptr) {
    event_lock_.unlock();
    return RPC_ERROR;
  }
  (void)events_.erase(tev->fd);

  // Don't delete tev immediately, let's push it into deleted_events_, before next epoll_wait,we will free
  // all events in deleted_events_.
  AddDeletedEvent(tev);

  event_lock_.unlock();
  ev.events = 0;
  ev.data.ptr = tev;

  ret = epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, &ev);
  if (ret < 0) {
    MS_LOG(ERROR) << "Failed to call delete fd in epoll, fd:" << fd << ",epollfd:" << epoll_fd_ << ",errno:" << errno;
    return RPC_ERROR;
  }
  return RPC_OK;
}

int EventLoop::UpdateEpollEvent(int fd, uint32_t events) {
  struct epoll_event ev;
  Event *tev = nullptr;
  int ret;

  tev = FindEvent(fd);
  if (tev == nullptr) {
    MS_LOG(ERROR) << "Failed to call event lookup, fd:" << fd << ",events:" << events_;
    return RPC_ERROR;
  }
  if (memset_s(&ev, sizeof(ev), 0, sizeof(ev)) != EOK) {
    MS_LOG(ERROR) << "Failed to call memset_s.";
    return RPC_ERROR;
  }

  ev.events = events;
  ev.data.ptr = tev;

  ret = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev);
  if (ret != 0) {
    MS_LOG(ERROR) << "Failed to modify fd in epoll, fd:" << fd << ",events:" << events << ",errno:" << errno;
    return RPC_ERROR;
  }
  return RPC_OK;
}

void EventLoop::AddDeletedEvent(Event *event) {
  if (event == nullptr) {
    return;
  }
  // caller need check eventData is not nullptr
  std::list<Event *> delete_event_list;

  // if fd not found, push eventData into deleted_events_[fd]
  std::map<int, std::list<Event *>>::iterator fdIter = deleted_events_.find(event->fd);
  if (fdIter == deleted_events_.end()) {
    deleted_events_[event->fd].push_back(event);
    return;
  }

  // if fd found, check if same eventData ptr exists
  delete_event_list = fdIter->second;
  std::list<Event *>::iterator eventIter = delete_event_list.begin();
  bool found = false;
  while (eventIter != delete_event_list.end()) {
    if (*eventIter == event) {
      MS_LOG(WARNING) << "The fd has been deleted before fd:" << event->fd << ",epoll_fd_:" << epoll_fd_;
      found = true;
      break;
    }
    ++eventIter;
  }

  // if found same eventptr, do nothing
  if (found) {
    return;
  }
  deleted_events_[event->fd].push_back(event);
  return;
}

void EventLoop::RemoveDeletedEvents() {
  std::map<int, std::list<Event *>>::iterator fdIter = deleted_events_.begin();

  while (fdIter != deleted_events_.end()) {
    std::list<Event *> delete_event_list = fdIter->second;
    std::list<Event *>::iterator eventIter = delete_event_list.begin();

    while (eventIter != delete_event_list.end()) {
      Event *deleteEv = *eventIter;
      delete deleteEv;
      deleteEv = nullptr;
      ++eventIter;
    }
    (void)deleted_events_.erase(fdIter++);
  }
  deleted_events_.clear();
}

int EventLoop::FindDeletedEvent(const Event *tev) {
  if (tev == nullptr) {
    return 0;
  }
  std::map<int, std::list<Event *>>::iterator fdIter = deleted_events_.find(tev->fd);
  if (fdIter == deleted_events_.end()) {
    return 0;
  }

  std::list<Event *> delete_event_list = fdIter->second;
  std::list<Event *>::iterator eventIter = delete_event_list.begin();

  while (eventIter != delete_event_list.end()) {
    if (*eventIter == tev) {
      return 1;
    }
    ++eventIter;
  }
  return 0;
}

void EventLoop::HandleEvent(struct epoll_event *events, size_t nevent) {
  if (events == nullptr) {
    return;
  }
  int found;
  Event *tev = nullptr;

  for (size_t i = 0; i < nevent; i++) {
    tev = reinterpret_cast<Event *>(events[i].data.ptr);

    if (tev != nullptr) {
      found = FindDeletedEvent(tev);
      if (found > 0) {
        MS_LOG(WARNING) << "The fd has been deleted from epoll fd:" << tev->fd << ",epoll_fd_:" << epoll_fd_;
        continue;
      }
      tev->handler(tev->fd, events[i].events, tev->data);
    }
  }
}

void EventLoop::Stop() {
  if (is_stop_) {
    return;
  }

  is_stop_ = true;
  uint64_t one = 1;

  auto retval = write(task_queue_event_fd_, &one, sizeof(one));
  if (retval <= 0 || retval != sizeof(one)) {
    MS_LOG(WARNING) << "Failed to write task_queue_event_fd_ fd:" << task_queue_event_fd_ << ",errno:" << errno;
  }
  return;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
