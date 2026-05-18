#!/usr/bin/env bash

# 这个脚本用于在 Linux 服务器后台启动模型推理服务.
# 它会自动进入项目根目录, 尝试激活 Conda 环境, 并把日志写入 logs/model-api.log.
# 默认监听 0.0.0.0:8000, 这样外部机器可以访问服务.

set -euo pipefail

# SCRIPT_DIR 表示当前脚本所在目录.
# PROJECT_ROOT 表示项目根目录, 也就是 api_server.py 所在目录.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 这些变量都支持通过环境变量覆盖.
# 例如: IQA_API_PORT=9000 ./scripts/start_model_api.sh
CONDA_ENV_NAME="${IQA_CONDA_ENV:-model-api}"
HOST="${IQA_API_HOST:-0.0.0.0}"
PORT="${IQA_API_PORT:-8000}"
LOG_DIR="${IQA_LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${IQA_LOG_FILE:-${LOG_DIR}/model-api.log}"
PID_FILE="${IQA_PID_FILE:-${LOG_DIR}/model-api.pid}"

mkdir -p "${LOG_DIR}"

# 如果 PID 文件存在, 先判断旧进程是否仍在运行.
# 这样可以避免重复启动多个模型服务, 否则会占用更多内存.
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}")"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "模型服务已经在运行, PID=${OLD_PID}"
        echo "日志文件: ${LOG_FILE}"
        exit 0
    fi

    # PID 文件存在但进程不存在, 说明是上次异常退出后留下的旧文件.
    rm -f "${PID_FILE}"
fi

cd "${PROJECT_ROOT}"

# Conda 在非交互脚本里通常不能直接 conda activate.
# 这里会先找到 conda.sh, 再激活环境.
# 如果你已经手动激活了环境, 或者不想让脚本激活 Conda, 可以设置 IQA_SKIP_CONDA=1.
if [[ "${IQA_SKIP_CONDA:-0}" != "1" ]]; then
    CONDA_SH=""

    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/root/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"
    fi

    if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV_NAME}"
    else
        echo "警告: 没有找到 conda.sh, 将使用当前 shell 的 python 启动服务." >&2
        echo "如果启动失败, 请先执行 conda activate ${CONDA_ENV_NAME}, 或设置正确的 Anaconda 路径." >&2
    fi
fi

# 使用 nohup 让服务在终端关闭后继续运行.
# 使用 >> 表示追加日志, 2>&1 表示把错误日志也写入同一个文件.
# 这里显式使用 uvicorn, 避免旧版本 api_server.py 只监听 localhost.
echo "[$(date '+%F %T')] 正在启动模型服务, host=${HOST}, port=${PORT}" >> "${LOG_FILE}"
nohup python -m uvicorn api_server:app --host "${HOST}" --port "${PORT}" >> "${LOG_FILE}" 2>&1 &

NEW_PID="$!"
echo "${NEW_PID}" > "${PID_FILE}"

# 稍等片刻后检查进程是否仍然存活.
# 如果模型加载失败, 进程通常会很快退出, 此时直接提示查看日志.
sleep 2
if kill -0 "${NEW_PID}" 2>/dev/null; then
    echo "模型服务已后台启动, PID=${NEW_PID}"
    echo "访问地址: http://${HOST}:${PORT}/health"
    echo "日志文件: ${LOG_FILE}"
    echo "停止命令: ./scripts/stop_model_api.sh"
else
    echo "模型服务启动失败, 请查看日志: ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
fi
