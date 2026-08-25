#!/bin/bash
# 打包设备上所有 rlog 文件为 tar.gz
# 保留目录结构（路线名/段号/rlog*），便于本地直接使用
# 使用方式：bash pack_all_rlogs.sh [输出路径]

set -e

REALDATA_DIR="/data/media/0/realdata"
DEFAULT_OUTPUT="/data/realdata_all_rlogs.tar.gz"

OUTPUT="${1:-$DEFAULT_OUTPUT}"

echo "============================================"
echo "  打包设备 rlog 文件"
echo "============================================"
echo "  数据目录: $REALDATA_DIR"
echo "  输出文件: $OUTPUT"
echo ""

# 检查数据目录
if [ ! -d "$REALDATA_DIR" ]; then
    echo "[ERROR] 数据目录不存在: $REALDATA_DIR"
    exit 1
fi

cd "$REALDATA_DIR"

# 统计信息
total_dirs=$(ls -d */ 2>/dev/null | wc -l)
total_rlogs=$(find . -maxdepth 2 -name "rlog*" -type f | wc -l)
total_size=$(find . -maxdepth 2 -name "rlog*" -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)

echo "  路线目录数: $total_dirs"
echo "  rlog 文件数: $total_rlogs"
echo "  rlog 总大小: $total_size"
echo ""

if [ "$total_rlogs" -eq 0 ]; then
    echo "[ERROR] 未找到 rlog 文件"
    exit 1
fi

# 确认
read -p "  确认打包? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "  打包中，请耐心等待..."

# 打包所有 rlog 文件，保留目录结构
# --ignore-failed-read: 忽略读取失败的文件
#   openpilot 运行时 loggerd 会清理旧 rlog，find 扫描后 tar 处理前可能已被删除
# || true: tar 对缺失文件返回非零状态，但其余文件已正常打包，豁免 set -e 中断
find . -maxdepth 2 -name "rlog*" -type f | tar -czf "$OUTPUT" --ignore-failed-read -T - || true

if [ -f "$OUTPUT" ]; then
    output_size=$(du -h "$OUTPUT" | cut -f1)
    packed_count=$(tar -tzf "$OUTPUT" 2>/dev/null | wc -l)
    skipped_count=$((total_rlogs - packed_count))
    echo ""
    echo "============================================"
    echo "  打包完成"
    echo "============================================"
    echo "  文件: $OUTPUT"
    echo "  大小: $output_size"
    echo "  统计 rlog: $total_rlogs  实际打包: $packed_count  跳过: $skipped_count"
    echo ""
    echo "  下载到本地后解压命令:"
    echo "    tar -xzf realdata_all_rlogs.tar.gz"
    echo ""
    echo "  然后用一键训练:"
    echo "    python nnlc_auto_train.py --data ./realdata --car <车型名>"
else
    echo "[ERROR] 打包失败"
    exit 1
fi
