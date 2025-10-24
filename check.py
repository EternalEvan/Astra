import torch
import os
import argparse
from collections import defaultdict
import time

def load_checkpoint(ckpt_path):
    """加载检查点文件"""
    if not os.path.exists(ckpt_path):
        return None
    
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        return state_dict
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None

def compare_parameters(state_dict1, state_dict2, threshold=1e-8):
    """比较两个状态字典的参数差异"""
    if state_dict1 is None or state_dict2 is None:
        return None
    
    updated_params = {}
    unchanged_params = {}
    
    for name, param1 in state_dict1.items():
        if name in state_dict2:
            param2 = state_dict2[name]
            
            # 计算参数差异
            diff = torch.abs(param1 - param2)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            if max_diff > threshold:
                updated_params[name] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'shape': param1.shape
                }
            else:
                unchanged_params[name] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'shape': param1.shape
                }
    
    return updated_params, unchanged_params

def categorize_parameters(param_dict):
    """将参数按类型分类"""
    categories = {
        'moe_related': {},
        'camera_related': {},
        'framepack_related': {},
        'attention': {},
        'other': {}
    }
    
    for name, info in param_dict.items():
        if any(keyword in name.lower() for keyword in ['moe', 'gate', 'expert', 'processor']):
            categories['moe_related'][name] = info
        elif any(keyword in name.lower() for keyword in ['cam_encoder', 'projector', 'camera']):
            categories['camera_related'][name] = info
        elif any(keyword in name.lower() for keyword in ['clean_x_embedder', 'framepack']):
            categories['framepack_related'][name] = info
        elif any(keyword in name.lower() for keyword in ['attn', 'attention']):
            categories['attention'][name] = info
        else:
            categories['other'][name] = info
    
    return categories

def print_category_summary(category_name, params, color_code=''):
    """打印某类参数的摘要"""
    if not params:
        print(f"{color_code}  {category_name}: 无参数")
        return
    
    total_params = len(params)
    max_diffs = [info['max_diff'] for info in params.values()]
    mean_diffs = [info['mean_diff'] for info in params.values()]
    
    print(f"{color_code}  {category_name} ({total_params} 个参数):")
    print(f"    最大差异范围: {min(max_diffs):.2e} ~ {max(max_diffs):.2e}")
    print(f"    平均差异范围: {min(mean_diffs):.2e} ~ {max(mean_diffs):.2e}")
    
    # 显示前5个最大变化的参数
    sorted_params = sorted(params.items(), key=lambda x: x[1]['max_diff'], reverse=True)
    print(f"    变化最大的参数:")
    for i, (name, info) in enumerate(sorted_params[:100]):
        shape_str = 'x'.join(map(str, info['shape']))
        print(f"      {i+1}. {name} [{shape_str}]: max_diff={info['max_diff']:.2e}")

def monitor_training(checkpoint_dir, check_interval=60):
    """监控训练过程中的参数更新"""
    print(f"🔍 开始监控训练进度...")
    print(f"📁 检查点目录: {checkpoint_dir}")
    print(f"⏰ 检查间隔: {check_interval}秒")
    print("=" * 80)
    
    previous_ckpt = None
    previous_step = -1
    
    while True:
        try:
            # 查找最新的检查点
            if not os.path.exists(checkpoint_dir):
                print(f"❌ 检查点目录不存在: {checkpoint_dir}")
                time.sleep(check_interval)
                continue
            
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('step') and f.endswith('.ckpt')]
            if not ckpt_files:
                print("⏳ 未找到检查点文件，等待中...")
                time.sleep(check_interval)
                continue
            
            # 按步数排序，获取最新的
            ckpt_files.sort(key=lambda x: int(x.replace('step', '').replace('.ckpt', '')))
            latest_ckpt_file = ckpt_files[-1]
            latest_ckpt_path = os.path.join(checkpoint_dir, latest_ckpt_file)
            
            # 提取步数
            current_step = int(latest_ckpt_file.replace('step', '').replace('.ckpt', ''))
            
            if current_step <= previous_step:
                print(f"⏳ 等待新的检查点... (当前: step{current_step})")
                time.sleep(check_interval)
                continue
            
            print(f"\n🔍 发现新检查点: {latest_ckpt_file}")
            
            # 加载当前检查点
            current_state_dict = load_checkpoint(latest_ckpt_path)
            if current_state_dict is None:
                print("❌ 无法加载当前检查点")
                time.sleep(check_interval)
                continue
            
            if previous_ckpt is not None:
                print(f"📊 比较 step{previous_step} -> step{current_step}")
                
                # 比较参数
                updated_params, unchanged_params = compare_parameters(
                    previous_ckpt, current_state_dict, threshold=1e-8
                )
                
                if updated_params is None:
                    print("❌ 参数比较失败")
                else:
                    # 分类显示结果
                    updated_categories = categorize_parameters(updated_params)
                    unchanged_categories = categorize_parameters(unchanged_params)
                    
                    print(f"\n✅ 已更新的参数 (总共 {len(updated_params)} 个):")
                    print_category_summary("MoE相关", updated_categories['moe_related'], '🔥')
                    print_category_summary("Camera相关", updated_categories['camera_related'], '📷')
                    print_category_summary("FramePack相关", updated_categories['framepack_related'], '🎞️')
                    print_category_summary("注意力相关", updated_categories['attention'], '👁️')
                    print_category_summary("其他", updated_categories['other'], '📦')
                    
                    print(f"\n⚠️  未更新的参数 (总共 {len(unchanged_params)} 个):")
                    print_category_summary("MoE相关", unchanged_categories['moe_related'], '❄️')
                    print_category_summary("Camera相关", unchanged_categories['camera_related'], '❄️')
                    print_category_summary("FramePack相关", unchanged_categories['framepack_related'], '❄️')
                    print_category_summary("注意力相关", unchanged_categories['attention'], '❄️')
                    print_category_summary("其他", unchanged_categories['other'], '❄️')
                    
                    # 检查关键组件是否在更新
                    critical_keywords = ['moe', 'cam_encoder', 'projector', 'clean_x_embedder']
                    critical_updated = any(
                        any(keyword in name.lower() for keyword in critical_keywords)
                        for name in updated_params.keys()
                    )
                    
                    if critical_updated:
                        print("\n✅ 关键组件正在更新！")
                    else:
                        print("\n❌ 警告：关键组件可能未在更新！")
                    
                    # 计算更新率
                    total_params = len(updated_params) + len(unchanged_params)
                    update_rate = len(updated_params) / total_params * 100
                    print(f"\n📈 参数更新率: {update_rate:.1f}% ({len(updated_params)}/{total_params})")
            
            # 保存当前状态用于下次比较
            previous_ckpt = current_state_dict
            previous_step = current_step
            
            print("=" * 80)
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
            break
        except Exception as e:
            print(f"❌ 监控过程中出错: {e}")
            time.sleep(check_interval)

def compare_two_checkpoints(ckpt1_path, ckpt2_path):
    """比较两个特定的检查点"""
    print(f"🔍 比较两个检查点:")
    print(f"  检查点1: {ckpt1_path}")
    print(f"  检查点2: {ckpt2_path}")
    print("=" * 80)
    
    # 加载检查点
    state_dict1 = load_checkpoint(ckpt1_path)
    state_dict2 = load_checkpoint(ckpt2_path)
    
    if state_dict1 is None or state_dict2 is None:
        print("❌ 无法加载检查点文件")
        return
    
    # 比较参数
    updated_params, unchanged_params = compare_parameters(state_dict1, state_dict2)
    
    if updated_params is None:
        print("❌ 参数比较失败")
        return
    
    # 分类显示结果
    updated_categories = categorize_parameters(updated_params)
    unchanged_categories = categorize_parameters(unchanged_params)
    
    print(f"\n✅ 已更新的参数 (总共 {len(updated_params)} 个):")
    for category_name, params in updated_categories.items():
        print_category_summary(category_name.replace('_', ' ').title(), params, '🔥')
    
    print(f"\n⚠️  未更新的参数 (总共 {len(unchanged_params)} 个):")
    for category_name, params in unchanged_categories.items():
        print_category_summary(category_name.replace('_', ' ').title(), params, '❄️')
    
    # 计算更新率
    total_params = len(updated_params) + len(unchanged_params)
    update_rate = len(updated_params) / total_params * 100
    print(f"\n📈 参数更新率: {update_rate:.1f}% ({len(updated_params)}/{total_params})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="检查模型参数更新情况")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe",
                       help="检查点目录路径")
    parser.add_argument("--compare", default=True, 
                       help="比较两个特定检查点，而不是监控")
    parser.add_argument("--ckpt1", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe/step1500_origin_cam_4.ckpt")
    parser.add_argument("--ckpt2", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe/step500_origin_cam_4.ckpt")
    parser.add_argument("--interval", type=int, default=60, 
                       help="监控检查间隔（秒）")
    parser.add_argument("--threshold", type=float, default=1e-8,
                       help="参数变化阈值")
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.ckpt1 or not args.ckpt2:
            print("❌ 比较模式需要指定 --ckpt1 和 --ckpt2")
        else:
            compare_two_checkpoints(args.ckpt1, args.ckpt2)
    else:
        monitor_training(args.checkpoint_dir, args.interval)