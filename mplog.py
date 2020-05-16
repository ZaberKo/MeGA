from logging import FileHandler
import multiprocessing, threading, logging, sys, traceback
import os
import logging.config


LOG_MASTER_RANK=0

class MPLog(object):
    def __init__(self,log_path,rank):
        self.rank=rank
        if self.rank==LOG_MASTER_RANK:
            # use line buffer
            dir_path=os.path.dirname(log_path)
            if len(dir_path)>0 and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            self.f=open(log_path,'w',encoding='utf-8',buffering=1)

    def __del__(self):
        if self.rank==LOG_MASTER_RANK:
            self.f.close()


    def log(self,msg):
        if self.rank==LOG_MASTER_RANK:
            self.f.write(str(msg)+'\n')