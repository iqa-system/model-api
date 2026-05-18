#!/usr/bin/env bash

# 这个脚本用于停止后台运行的模型推理服务.
# 它会读取 logs/model-api.pid, 并只停止命令行里包含 api_server.py 或 api_server:app 的进程.
# 这样可以降低 PID 被复用后误杀其它进程的风险.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${IQA_LOG_DIR:-${PROJECT_ROOT}/logs}"
PID_FILE="${IQA_PID_FILE:-${LOG_DIR}/model-api.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "没有找到 PID 文件: ${PID_FILE}"
    echo "服务可能没有通过 start_model_api.sh 启动, 或者已经停止."
    exit 0
fi

PID="$(cat "${PID_FILE}")"
if [[ -z "${PID}" ]]; then
    echo "PID 文件为空, 已删除: ${PID_FILE}"
    rm -f "${PID_FILE}"
    exit 0
fi

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "PID=${PID} 对应的进程不存在, 已清理旧 PID 文件."
    rm -f "${PID_FILE}"
    exit 0
fi

# 读取进程命令行, 并确认它确实是当前项目的模型服务.
# /proc 是 Linux 提供的进程信息目录.
CMDLINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null || true)"
if [[ "${CMDLINE}" != *"api_server.py"* && "${CMDLINE}" != *"api_server:app"* ]]; then
    echo "拒绝停止 PID=${PID}, 因为它看起来不是模型服务进程."
    echo "实际命令行: ${CMDLINE}"
    echo "如果你确认要停止它, 请手动执行: kill ${PID}"
    exit 1
fi

echo "正在停止模型服务, PID=${PID}"
kill "${PID}"

# 等待进程优雅退出.
# 大模型服务释放内存可能需要几秒钟, 所以这里最多等 20 秒.
for _ in $(seq 1 20); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        echo "模型服务已停止."
        exit 0
    fi
    sleep 1
done

echo "模型服务没有在 20 秒内退出."
echo "如果确认需要强制停止, 可以执行: kill -9 ${PID}"
exit 1
