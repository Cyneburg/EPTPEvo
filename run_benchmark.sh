#!/bin/bash

# EPTPEvo Benchmark 批量运行脚本
# 使用方法: bash run_benchmark.sh

set -e  # 遇到错误立即退出

# ============================================================
# 参数解析
# 用法:
#   bash run_benchmark.sh                                      # 运行全部数据集
#   bash run_benchmark.sh <dataset_name>                       # 运行单个数据集
#   bash run_benchmark.sh <dataset1> <dataset2> ...            # 运行多个数据集
#   bash run_benchmark.sh --force <dataset_name>               # 强制重新运行单个数据集
#   bash run_benchmark.sh --force <dataset1> <dataset2> ...    # 强制重新运行多个数据集
# ============================================================
FORCE=false
DATASET_LIST=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        -*)
            echo "未知选项: $1"
            exit 1
            ;;
        *)
            DATASET_LIST+=("$1")
            shift
            ;;
    esac
done

# 配置路径
BENCHMARK_DIR="/Benchmark"
OUTPUT_DIR="/result/"
EPTPEvo_DIR="/public/home/EPTPEvo"
DATASET_JSON="/dataset.json"

# 从dataset.json读取数据集（支持单个、多个或全部）
if [ ${#DATASET_LIST[@]} -gt 0 ]; then
    DATASETS="${DATASET_LIST[*]}"
    echo "运行指定数据集: ${DATASET_LIST[*]}"
    if [ "$FORCE" = true ]; then
        echo "强制模式: 将删除已有结果重新运行"
    fi
else
    DATASETS=$(python3 -c "import json; print('\n'.join([d['file'].replace('.csv','') for d in json.load(open('$DATASET_JSON'))]))")
fi

# 统计信息
total=$(echo "$DATASETS" | wc -l)
current=0
success=0
failed=0

echo "=========================================="
echo "EPTPEvo Benchmark Runner"
echo "=========================================="
echo "Total datasets: $total"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 遍历每个数据集
for dataset in $DATASETS; do
    current=$((current + 1))
    echo ""
    echo "[$current/$total] Processing: $dataset"
    echo "------------------------------------------"

    # 检查benchmark文件是否存在
    if [ ! -f "$BENCHMARK_DIR/${dataset}.csv" ]; then
        echo "WARNING: Benchmark file not found: $BENCHMARK_DIR/${dataset}.csv"
        failed=$((failed + 1))
        continue
    fi

    # 检查是否已经处理过（--force模式下删除已有结果）
    if [ -d "$OUTPUT_DIR/$dataset" ] && [ "$(ls -A $OUTPUT_DIR/$dataset 2>/dev/null)" ]; then
        if [ "$FORCE" = true ]; then
            echo "FORCE: 删除已有结果 $OUTPUT_DIR/$dataset"
            rm -rf "$OUTPUT_DIR/$dataset"
        else
            echo "SKIP: Results already exist for $dataset"
            success=$((success + 1))
            continue
        fi
    fi

    # 1. 创建临时目录
    temp_dir="$EPTPEvo_DIR/data/temp_${dataset}"
    temp_subdir="$temp_dir/substitutions"
    mkdir -p "$temp_subdir"
    cp "$BENCHMARK_DIR/${dataset}.csv" "$temp_subdir/"

    # 2. 修改config.json指向临时目录
    # 备份原始config
    cp "$EPTPEvo_DIR/fsfp/config.json" "$EPTPEvo_DIR/fsfp/config.json.bak"

    # 创建临时config
    python3 -c "
import json
with open('$EPTPEvo_DIR/fsfp/config.json', 'r') as f:
    config = json.load(f)
config['raw_data_dir'] = '$temp_subdir'
config['data_path'] = '$temp_dir/merged.pkl'
with open('$EPTPEvo_DIR/fsfp/config.json', 'w') as f:
    json.dump(config, f, indent=4)
"

    # 3. 运行preprocess.py
    echo "Running preprocess.py..."
    cd "$EPTPEvo_DIR"
    python preprocess.py -s

    # 4. 运行main.py（使用esm1b模型）
    echo "Running main.py with esm1b..."
    python main.py --model esm1b --train_data_size 1.0 --train_size 12 --train_batch 12 --max_al_rounds 10 --eval_batch 12

    # 5. 恢复原始config
    mv "$EPTPEvo_DIR/fsfp/config.json.bak" "$EPTPEvo_DIR/fsfp/config.json"

    # 6. 清理临时文件
    echo "Cleaning up temporary files..."
    rm -rf "$temp_dir"

    # 7. 检查结果是否生成
    if [ -d "$OUTPUT_DIR/$dataset" ] && [ "$(ls -A $OUTPUT_DIR/$dataset 2>/dev/null)" ]; then
        echo "SUCCESS: Results saved to $OUTPUT_DIR/$dataset/"
        success=$((success + 1))
    else
        echo "ERROR: No results generated for $dataset"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Total: $total"
echo "Success: $success"
echo "Failed: $failed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
