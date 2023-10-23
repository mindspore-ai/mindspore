# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore import log as logger
from mindspore._extends.parallel_compile.akg_compiler.akg_process import create_akg_parallel_process, \
    create_akg_v2_parallel_process


class Messager:

    '''Messager'''

    def __init__(self, fdin, fdout):
        self.fdin = fdin
        self.fdout = fdout
        self.fin = os.fdopen(fdin, "r")
        self.fout = os.fdopen(fdout, "w")
        self.message = ''

    def __del__(self):
        os.close(self.fdin)
        os.close(self.fdout)

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
                logger.debug("[TRACE] read nothing...")
                self.exit()
            if res[len(res) - 1] == '\n':
                res = res[0:len(res) - 1]
            self.message = res
            logger.debug(f"[IN] {self.message}")
        except (EOFError, KeyboardInterrupt):
            self.exit()
        finally:
            pass
        if self.message == '' or self.message == 'FINISH':
            self.send_ack()
            self.exit()
        return self.message

    def send_res(self, res, keep_format=True):
        """
        Send result to remote

        Args:
            keep_format: True or False
        """
        logger.debug(f"[OUT] {str(res)}")
        if keep_format:
            res_str = str(res).replace('\n', '[LF]').replace('\r', '[CR]').replace(' ', '[SP]')
        else:
            res_str = str(res).replace('\n', '').replace('\r', '').replace(' ', '')
        tag = '[~]'  # The same as client kTAG

        # Not write by print(tag + res_str, flush=True) any more
        try:
            self.fout.write(tag + res_str + "\n")
            self.fout.flush()
        except BrokenPipeError as err:
            logger.info(f"[TRACE] Write {str(err)}")
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

    def loop(self):
        """
        Messaging loop
        """
        while True:
            self.handle()

    def run(self):
        self.loop()

    def handle(self):
        """
        A interface communicates with remote.

        Note:
            All subclasses should override this interface.
        """
        raise NotImplementedError

    def exit(self):
        """
        A interface handles the procedure before exit.

        Note:
            All subclasses should override this interface.
        """
        raise NotImplementedError


class AkgBuilderBase():
    """Base Class for kernel compiler building wrapper"""

    def __init__(self, platform):
        self.platform = platform
        self.attrs = None
        self.akg_processor = None

    def create(self, process_num, waitime):
        """ Create compiler processor"""
        del process_num, waitime
        raise NotImplementedError

    def accept_json(self, json):
        """ Accept json"""
        return self.akg_processor.accept_json(json)

    def compile(self):
        """Compile"""
        return self.akg_processor.compile(self.attrs)

    def handle(self, messager, arg):
        """Handle message about compiler"""
        if arg == 'AKG/START':
            messager.send_ack()
            process_num_str = messager.get_message()
            messager.send_ack()
            wait_time_str = messager.get_message()
            messager.send_ack()
            self.create(int(process_num_str), int(wait_time_str))
        elif arg == 'AKG/ATTR':
            messager.send_ack()
            self.attrs = messager.get_message()
            messager.send_ack()
        elif arg == 'AKG/DATA':
            messager.send_ack()
            while True:
                req = messager.get_message()
                if req.startswith('{'):
                    self.accept_json(req)
                    messager.send_ack()
                elif req == 'AKG/WAIT':
                    res = self.compile()
                    messager.send_res(res)
                    break
                else:
                    messager.send_ack(False)
                    break
        else:
            raise RuntimeError(f"Unknown message type: {arg}")


class AkgBuilder(AkgBuilderBase):
    """Akg building wrapper"""

    def create(self, process_num, waitime):
        """ Create akg processor"""
        self.akg_processor = create_akg_parallel_process(process_num, waitime, self.platform)


class AkgV2Builder(AkgBuilderBase):
    """Akg V2 building wrapper"""

    def create(self, process_num, waitime):
        """ Create akg v2 processor"""
        self.akg_processor = create_akg_v2_parallel_process(process_num, waitime, self.platform)


def get_logger():
    return logger
