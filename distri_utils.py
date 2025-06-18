import socket
import os
import psutil
import torch
import torch.distributed as dist
import datetime
import signal
import sys

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    print(f"Rank {rank}: 初始化进程组，PID={os.getpid()}, PORT={port}")
    try:
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(minutes=20),  # 增加到5分钟
            device_id=torch.device(f'cuda:{rank}')
        )
        
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: 进程组初始化成功，使用 GPU {rank}")
    except Exception as e:
        print(f"Rank {rank}: 进程组初始化失败 - {str(e)}")
        raise

def cleanup():
    if dist.is_initialized():
        print(f"PID {os.getpid()}: 销毁进程组")
        dist.destroy_process_group()
    else:
        print(f"PID {os.getpid()}: 进程组未初始化，跳过销毁")

def kill_child_processes(timeout=3):
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    
    if not children:
        return
    
    for proc in children:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    
    # 等待子进程终止
    _, alive = psutil.wait_procs(children, timeout=timeout)
    
    # 如果仍有存活的子进程，强制结束它们
    for proc in alive:
        try:
            print(f"强制结束子进程: {proc.pid}")
            proc.kill()
        except psutil.NoSuchProcess:
            pass

def launch_process(rank):
    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, 'getppid') else None
    print(f"启动进程: Rank {rank}, PID={pid}, PPID={ppid}")
    
    def handle_sigterm(signum, frame):
        print(f"Rank {rank} (PID {pid}): 收到信号 {signum}, 忽略...")
    
    def handle_sigint(signum, frame):
        print(f"Rank {rank} (PID {pid}): 收到中断信号 {signum}, 清理资源...")
        if dist.is_initialized():
            cleanup()
        kill_child_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)