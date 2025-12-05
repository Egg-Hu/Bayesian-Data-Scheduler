#!/bin/bash

# 节点和所属分区映射（从你提供的表格整理）
declare -A node_partitions=(
  [node01]=NA100q
  [node02]=PA100q
  [node03]=PA100q
  [node04]=PA100q
  [node05]=PA40q
  [node06]=PH100q
  [node07]=PA40q
  [node08]=RTXA6Kq
  [node09]=RTXA6Kq
  [node10]=RTXA6Kq
  [node11]=RTXA6Kq
  [node12]=PA100q
  [node13]=NA100q
  [node14]=HPCq
  [node15]=NH100q
)

echo "================== GPU 显存使用情况总览 =================="

for node in "${!node_partitions[@]}"; do
  partition=${node_partitions[$node]}
  echo -e "\n🔹 节点：$node  （分区：$partition）"
  echo "----------------------------------------------------------"
  echo -e "GPU   Name                       Used(MB)  Total(MB)"

  # 使用 timeout 限制 srun 最多运行 10 秒，防止卡死
  timeout 10s srun -p "$partition" -N1 -w "$node" --gres=gpu:1 --ntasks=1 --quiet bash -c \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null | \
  while IFS=',' read -r index name used total; do
    # 去除空格
    index=$(echo "$index" | sed 's/^ *//;s/ *$//')
    name=$(echo "$name" | sed 's/^ *//;s/ *$//')
    used=$(echo "$used" | sed 's/^ *//;s/ *$//')
    total=$(echo "$total" | sed 's/^ *//;s/ *$//')
    printf "%-5s %-25s %-10s %-10s\n" "$index" "$name" "$used" "$total"
  done

  # 如果上一条命令失败，则显示警告
  if [ $? -ne 0 ]; then
    echo "⚠️  无法连接 $node，节点可能忙/异常/无 GPU 或权限问题"
  fi
done



# example
# srun -p HPCq -w node14 -n 1 --gres=gpu:2 -t 2-12:00:00 --pty bash
# srun -p PA100q -w node03 -n 1 --gres=gpu:1 -t 6-12:00:00 --pty bash
# srun -p RTXA6Kq -w node11 -n 1 --gres=gpu:1 -t 4-12:00:00 --pty bash
# srun -p NA100q -w node01 -n 1 --gres=gpu:1 -t 6-12:00:00 --pty bash
