#!/usr/bin/env bash

# 这个脚本用于查看模型推理服务的后台运行状态.
# 它会读取 PID 文件, 检查进程是否存在, 并展示端口监听和最近日志.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="${IQA_API_PORT:-8000}"
LOG_DIR="${IQA_LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${IQA_LOG_FILE:-${LOG_DIR}/model-api.log}"
PID_FILE="${IQA_PID_FILE:-${LOG_DIR}/model-api.pid}"

if [[ -f "${PID_FILE}" ]]; then
    PID="$(cat "${PID_FILE}")"
    if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
        echo "模型服务正在运行, PID=${PID}"
        echo "进程命令:"
        tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null || true
        echo
    else
        echo "PID 文件存在, 但进程不存在: ${PID_FILE}"
    fi
else
    echo "没有找到 PID 文件, 服务可能没有通过 start_model_api.sh 启动."
fi

echo
echo "端口监听情况:"
if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | grep ":${PORT}" || echo "没有发现 ${PORT} 端口监听."
else
    echo "当前系统没有 ss 命令, 可安装 iproute2 后再查看端口监听."
fi

echo
echo "最近日志:"
if [[ -f "${LOG_FILE}" ]]; then
    tail -n 30 "${LOG_FILE}"
else
    echo "没有找到日志文件: ${LOG_FILE}"
fi
