import torch
import time
import argparse
from threading import Thread

def gpu_worker(gpu_id, duration, tensor_size):
    """单个GPU的工作线程，负责持续进行张量运算"""
    try:
        # 设置当前线程使用的GPU
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        # 打印GPU信息
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU {gpu_id} 启动: {gpu_name}")
        
        # 创建大随机张量
        tensor_a = torch.randn(tensor_size, tensor_size, device=device)
        tensor_b = torch.randn(tensor_size, tensor_size, device=device)
        
        # 预热GPU
        for _ in range(10):
            result = torch.matmul(tensor_a, tensor_b)
            torch.cuda.synchronize(device)
        
        # 开始持续运算
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration:
            # 矩阵乘法运算
            result = torch.matmul(tensor_a, tensor_b)
            
            # 定期更新张量避免优化
            if iterations % 100 == 0:
                tensor_a = 0.999 * tensor_a + 0.001 * torch.randn_like(tensor_a)
                tensor_b = 0.999 * tensor_b + 0.001 * torch.randn_like(tensor_b)
            
            iterations += 1
            
            # 每10秒打印一次状态
            if iterations % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"GPU {gpu_id}: 已运行 {elapsed:.1f} 秒, 完成 {iterations} 次迭代")
            
            # 短暂同步确保计算完成
            if iterations % 100 == 0:
                torch.cuda.synchronize(device)
        
        # 计算结束统计
        elapsed = time.time() - start_time
        print(f"GPU {gpu_id} 完成: 总时间 {elapsed:.1f} 秒, 总迭代 {iterations} 次, "
              f"平均每秒 {iterations/elapsed:.2f} 次")
        
    except Exception as e:
        print(f"GPU {gpu_id} 出错: {str(e)}")
    
    finally:
        # 清理内存
        torch.cuda.empty_cache()

def multi_gpu_stress_test(duration, tensor_size, use_gpus=None):
    """多GPU压力测试主函数"""
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("错误: 未检测到可用GPU")
        return
    
    # 确定要使用的GPU
    if use_gpus is None:
        use_gpus = list(range(available_gpus))
    else:
        # 验证GPU ID有效性
        use_gpus = [g for g in use_gpus if 0 <= g < available_gpus]
        if not use_gpus:
            print("错误: 没有有效的GPU ID")
            return
    
    print(f"检测到 {available_gpus} 张GPU，将使用 {len(use_gpus)} 张: {use_gpus}")
    
    # 为每张GPU创建并启动线程
    threads = []
    for gpu_id in use_gpus:
        thread = Thread(target=gpu_worker, args=(gpu_id, duration, tensor_size))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("所有GPU测试完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多GPU压力测试程序')
    parser.add_argument('--duration', type=int, default=6000000, 
                      help='测试持续时间(秒)，默认60秒')
    parser.add_argument('--size', type=int, default=4096, 
                      help='每张GPU上的张量大小，默认4096x4096')
    parser.add_argument('--gpus', type=int, nargs='+', 
                      help=f'指定要使用的GPU ID，如 --gpus 0 1 2 3 4 5 6 7')
    args = parser.parse_args()
    
    # 运行多GPU测试
    multi_gpu_stress_test(args.duration, args.size, args.gpus)
