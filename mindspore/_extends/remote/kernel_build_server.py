# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""kernel build server"""
import os
import sys
import time
from mindspore._extends.parallel_compile.tbe_compiler.tbe_process import create_tbe_parallel_process, op_select_format, check_supported
from mindspore._extends.parallel_compile.akg_compiler.akg_process import create_akg_parallel_process

class TbeBuilder:
    """Tbe building wrapper"""

    def __init__(self):
        self.tbe_builder = create_tbe_parallel_process()

    def start(self, json):
        return self.tbe_builder.start_compile_op(json)

    def wait(self):
        return self.tbe_builder.wait_one()

    def reset(self):
        self.tbe_builder.reset_task_info()

    def exit(self):
        self.tbe_builder.exit()

class AkgBuilder:
    """Akg building wrapper"""

    def __init__(self):
        pass

    def create(self, process_num, waitime):
        self.akg_builder = create_akg_parallel_process(process_num, waitime)

    def accept_json(self, json):
        return self.akg_builder.accept_json(json)

    def compile(self):
        return self.akg_builder.compile()

class Messager:
    '''Messager'''

    def __init__(self):
        logger.info('[TRACE]', 'Messager init...')
        self.message = ''
        self.tbe_builder = TbeBuilder()
        self.akg_builder = AkgBuilder()

    def get_message(self):
        """
        Get message from remote

        Returns:
            message
        """
        try:
            # Not read by input() anymore
            res = self.fin.readline()
            if not res:
                logger.info('[TRACE]', "read <empty>")
                self.exit()
            if res[len(res) - 1] == '\n':
                res = res[0:len(res)-1]
            self.message = res
            logger.debug('[IN]', self.message)
        except EOFError:
            self.exit()
        finally:
            pass
        if self.message == '' or self.message == 'FIN':
            self.send_ack()
            self.exit()
        return self.message

    def send_res(self, res, keep_format=True):
        """
        Send result to remote

        Args:
            keep_format: True or False
        """
        logger.debug('[OUT]', str(res))
        if keep_format:
            res_str = str(res).replace('\n', '[LF]').replace('\r', '[CR]').replace(' ', '[SP]')
        else:
            res_str = str(res).replace('\n', '').replace('\r', '').replace(' ', '')
        tag = '[~]' # The same as client kTAG

        # Not write by print(tag + res_str, flush=True) any more
        try:
            self.fout.write(tag + res_str + "\n")
            self.fout.flush()
        except BrokenPipeError as err:
            logger.info('[TRACE]', 'Write, ' + str(err))
            self.exit()
        finally:
            pass

    def send_ack(self, success=True):
        """
        Send ack to remote

        Args:
            success: True or False
        """
        if success:
            self.send_res('ACK')
        else:
            self.send_res('ERR')

    def handle(self):
        """
        Communicate with remote
        """
        arg = self.get_message()
        if arg == 'TBE/START':
            self.send_ack()
            json = self.get_message()
            res = self.tbe_builder.start(json)
            self.send_res(res)
        elif arg == 'TBE/WAIT':
            self.send_ack()
            task_id, res, pre = self.tbe_builder.wait()
            logger.debug('[TRACE]', str(task_id) + '/' + str(res) + '/' + str(pre))
            if self.get_message() != 'CONT':
                self.send_ack(False)
                self.exit()
            self.send_res(task_id)
            if self.get_message() != 'CONT':
                self.send_ack(False)
                self.exit()
            self.send_res(res)
            if self.get_message() != 'CONT':
                self.send_ack(False)
                self.exit()
            self.send_res(pre)
        elif arg == 'TBE/RESET':
            self.tbe_builder.reset()
            self.send_ack()
        elif arg == 'AKG/START':
            self.send_ack()
            process_num_str = self.get_message()
            self.send_ack()
            wait_time_str = self.get_message()
            self.akg_builder.create(int(process_num_str), int(wait_time_str))
            self.send_ack()
        elif arg == 'AKG/DATA':
            self.send_ack()
            while True:
                req = self.get_message()
                if req.startswith('{'):
                    self.akg_builder.accept_json(req)
                    self.send_ack()
                elif req == 'AKG/WAIT':
                    res = self.akg_builder.compile()
                    self.send_res(res)
                    break
                else:
                    self.send_ack(False)
                    break
        elif arg == 'FORMAT':
            self.send_ack()
            json = self.get_message()
            self.send_res(op_select_format(json))
        elif arg == 'SUPPORT':
            self.send_ack()
            json = self.get_message()
            logger.debug('[SUPPORT]', json)
            try:
                res = check_supported(json)
            except json.decoder.JSONDecodeError:
                self.send_ack(False)
                self.exit()
            finally:
                pass
            self.send_res(res)
        else:
            self.send_ack(False)
            self.exit()

    def loop(self):
        """
        Messaging loop
        """
        while True:
            self.handle()

    def exit(self):
        os.close(self.fdin)
        os.close(self.fdout)
        self.tbe_builder.reset()
        self.tbe_builder.exit()
        logger.info('[TRACE]', 'Messager Exit...')
        exit()

    def run(self, fdin, fdout):
        self.fdin = fdin
        self.fdout = fdout
        self.fin = os.fdopen(fdin, "r")
        self.fout = os.fdopen(fdout, "w")
        self.loop()

class Logger:
    """
    Replace dummy 'logger' to output log as below:
    logger = Logger(0, True, "remote_kernel_build_" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".log")
    """
    def __init__(self, level=1, dumpfile=False, filename='Logger.log'):
        """
        Args:
            level: 0 for debug and info, 1 for info
            dumpfile: if dump log into file
        """
        self.level = level
        self.dumpfile = dumpfile
        if self.dumpfile:
            self.log = open(filename, "a")

    def write(self, msg):
        self.log.write(msg)
        self.flush()

    def writeline(self, tag, msg):
        prefix = tag + ' REMOTE(' + str(os.getpid()) + ',python)'
        line = prefix + '\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ':\t' + msg
        print(line, flush=True)
        if self.dumpfile:
            self.write(line + '\n')

    def debug(self, tag, msg):
        if self.level == 0:
            self.writeline('[DEBUG]' + tag, msg)

    def info(self, tag, msg):
        self.writeline('[INFO]' + tag, msg)

    def flush(self):
        self.log.flush()

class DummyLogger:
    """DummyLogger"""
    def __init__(self):
        pass

    def debug(self, tag, msg):
        pass

    def info(self, tag, msg):
        pass

logger = DummyLogger()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Incorrect argv: {}'.format(sys.argv))
    logger.debug('[TRACE]', 'argv: ' + str(sys.argv))
    messager = Messager()
    messager.run(int(sys.argv[1]), int(sys.argv[2]))
