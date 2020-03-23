from logging import FileHandler
import multiprocessing, threading, logging, sys, traceback
import os
import logging.config
import yaml

def init_log(filename=None):
    with open('logging.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
    if filename is not None:
        config['handlers']['mplog']['filename']=filename
    logging.config.dictConfig(config)


class MultiProcessingHandler(logging.Handler):
    def __init__(self, filename, mode):
        logging.Handler.__init__(self)
        self._handler = FileHandler(filename, mode)
        self.queue = multiprocessing.Queue(-1)

        t = threading.Thread(target=self.receive)
        t.daemon = True
        t.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
                # print('received on pid {}'.format(os.getpid()))
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

        self.queue.close()
        self.queue.join_thread()
    
    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args have been stringified. Removes any
        # chance of unpickleable things inside and possibly reduces message size
        # sent over the pipe
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)