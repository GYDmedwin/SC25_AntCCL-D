#!/usr/bin/python
#****************************************************************#
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2024-10-21 10:21
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2024-10-21 10:21
# Function: 
#***************************************************************#

# -*- coding: utf-8 -*-


import os
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf
from pypai.conf import KMConf
from pypai.conf import GpuType
from aistudio_common.openapi.models.data_store import DataStore

def torch_train():
    km_conf = KMConf(
        # image="reg.docker.alibaba-inc.com/aii/aistudio:9850129-20241020120833"
        # image="reg.docker.alibaba-inc.com/aii/aistudio:9850129-20241021112249"
        image="reg.docker.alibaba-inc.com/aii/aistudio:9850129-20241028162359"

    )
    command = """mkdir /hetero_infer && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-49-xqs71.cn-heyuan-alipay.nas.aliyuncs.com:/ /hetero_infer && \
pip install datasets==2.15.0 evaluate==0.4.0 peft==0.3.0 && \
export NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,COLL && \
export XCCL_ENABLE_LOG_TIME=1 && \
export XCCL_DIAGNOSE_MODE=1 && \
bash llama2_7b.sh /output_dir"""
    worker_num = 1
    master = ExecConf(cpu=60, memory=819200, gpu_num=8, num=1, shared_memory=204800, gpu_type="h20", disk_m=102400)
    worker = ExecConf(cpu=60, memory=819200, gpu_num=8, num=worker_num, shared_memory=204800, gpu_type="h20", disk_m=102400)
    job = PythonJobBuilder(source_root="./",
                           km_conf=km_conf,
                           command=command,
                           master=master,
                           worker=worker,
                           runtime="pytorch",
                           rdma=True,
                           envs={
                            "LD_LIBRARY_PATH":"/home/admin/nccl/lib:/usr/lib64:/usr/local/cuda/lib64",
                           },
                          )

    job.run()

if __name__ == "__main__":
    torch_train()
