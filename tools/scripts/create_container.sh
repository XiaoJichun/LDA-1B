#!/bin/bash
set -eo pipefail

# ====================== 单容器配置 ======================
ORGI_SERVICE="orgi_services"

NEW_SERVICE_NAME="lda_1b"
IMAGE_NAME="lda_1b"
CONTAINER_NAME="lda_1b"

WORKING_DIR="/root/workspace"
VOLUME1="~/workspace/:/root/workspace"
VOLUME2="/mnt/inaisfs/manip-asset-gpfs/writeable-dir/vla_repo:/root/workspace/vla_repo"
PROXY_SCRIPT_NAME="create_proxy.sh"
# =========================================================

# 自动找目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROXY_SCRIPT="$SCRIPT_DIR/$PROXY_SCRIPT_NAME"

echo -e "\033[32m[INFO] 脚本启动成功，开始执行...\033[0m"

# ====================== 模式选择 ======================
MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo -e "\n\033[36m请选择安装方式：\033[0m"
    echo -e "  \033[33m[1]\033[0m  国内镜像模式 —— Docker基础镜像/apt/pip/conda 全部使用国内镜像源，无需代理"
    echo -e "  \033[33m[2]\033[0m  原始模式     —— 原始安装方式，自动检测并启用代理"
    read -rp $'\n请输入选项 [1/2]（默认 2）: ' MODE
    MODE="${MODE:-2}"
fi

case "$MODE" in
    1) echo -e "\033[32m✅ 已选择：国内镜像模式\033[0m"
       DOCKERFILE="tools/docker/Dockerfile.cn" ;;
    2) echo -e "\033[32m✅ 已选择：原始模式\033[0m"
       DOCKERFILE="tools/docker/Dockerfile" ;;
    *) echo -e "\033[31m❌ 无效选项：$MODE，请输入 1 或 2\033[0m"; exit 1 ;;
esac
# ======================================================

# 找到原 docker-compose.yml
COMPOSE_FILE=""
SEARCH_DIR="$SCRIPT_DIR"
for _ in {1..10}; do
    if [ -f "$SEARCH_DIR/tools/docker/docker-compose.yml" ]; then
        COMPOSE_FILE="$SEARCH_DIR/tools/docker/docker-compose.yml"
        break
    fi
    SEARCH_DIR=$(dirname "$SEARCH_DIR")
done

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "\033[31m❌ 找不到 docker-compose.yml\033[0m"
    exit 1
fi

# 关键：进入 compose 所在目录，解决相对路径问题
COMPOSE_DIR=$(dirname "$COMPOSE_FILE")
cd "$COMPOSE_DIR" || exit 1
echo -e "\033[32m✅ 进入构建目录：$COMPOSE_DIR\033[0m"

# ===================== 代理 / 网络处理 =====================
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

check_url() {
    local url="$1"
    local name="$2"
    if curl -s --connect-timeout 5 --head "$url" | head -n 1 | grep -qE "200|301|302"; then
        echo -e "\033[32m✅ $name 连通正常\033[0m"
        return 0
    else
        echo -e "\033[31m❌ $name 连通失败\033[0m"
        return 1
    fi
}

# ---- shell 层代理处理（仅模式 2）----
if [ "$MODE" = "2" ]; then
    echo -e "\n\033[34m=== 清空代理，裸连检测网络 ===\033[0m"
    GITHUB_OK=0
    GOOGLE_OK=0
    check_url "https://github.com" "GitHub" && GITHUB_OK=1
    check_url "https://google.com" "Google" && GOOGLE_OK=1

    if [ $GITHUB_OK -eq 1 ] && [ $GOOGLE_OK -eq 1 ]; then
        echo -e "\n\033[32m🎉 裸连网络正常，不启用代理\033[0m"
    else
        echo -e "\n\033[33m⚠️  裸连失败，自动启用代理...\033[0m"
        source "$PROXY_SCRIPT" on
        echo -e "\n\033[34m=== 代理已启用，重新检测 ===\033[0m"
        check_url "https://github.com" "GitHub"
        check_url "https://google.com" "Google"
    fi
else
    echo -e "\n\033[34m=== 国内镜像模式：跳过 shell 代理检测 ===\033[0m"
fi

# ---- Docker 守护进程代理检测（两种模式均执行）----
# 守护进程代理独立于 shell 环境，不可用时会拦截所有 docker pull，导致构建失败
echo -e "\n\033[34m=== 检测 Docker 守护进程代理 ===\033[0m"
echo -ne "\033[33m正在查询 Docker 守护进程信息...\033[0m"
DOCKER_DAEMON_PROXY=$(timeout 10 docker system info 2>/dev/null | grep "HTTP Proxy:" | awk '{print $3}' || true)
echo -e "\r\033[K"  # 清除"正在查询"提示行
if [ -n "$DOCKER_DAEMON_PROXY" ] && [ "$DOCKER_DAEMON_PROXY" != "(null)" ]; then
    D_HOST=$(echo "$DOCKER_DAEMON_PROXY" | sed 's|.*://||' | cut -d: -f1)
    D_PORT=$(echo "$DOCKER_DAEMON_PROXY" | sed 's|.*://||' | cut -d: -f2 | cut -d/ -f1)
    if (echo >/dev/tcp/"${D_HOST}"/"${D_PORT}") 2>/dev/null; then
        echo -e "\033[32m✅ Docker 守护进程代理 ${DOCKER_DAEMON_PROXY} 可用\033[0m"
    else
        echo -e "\033[31m❌ Docker 守护进程代理 ${DOCKER_DAEMON_PROXY} 不可用\033[0m"
        echo -e "\033[33m   该代理会拦截所有 docker pull，请先修复再运行本脚本\033[0m"
        echo -e "\033[36m\n修复方案（二选一）：\033[0m"
        echo -e "  方案A  启动本地代理程序（Clash/V2Ray 等），然后重新运行本脚本"
        echo -e "  方案B  移除 Docker 守护进程代理（注意：需要重启 Docker 服务）："
        echo -e "           sudo rm -f /etc/systemd/system/docker.service.d/http-proxy.conf"
        echo -e "           sudo systemctl daemon-reload && sudo systemctl restart docker"
        exit 1
    fi
else
    echo -e "\033[32m✅ Docker 守护进程未配置代理\033[0m"
fi
# ===========================================================

# ===================== 安全创建临时文件 =====================
# 注意：compose 的相对路径是相对 compose 文件位置解析的，
# 临时文件必须放在同目录，否则 build context/dockerfile 会失效。
TMP_COMPOSE="${COMPOSE_DIR}/.temp_docker-compose-$(date +%s).yml"
cp "$COMPOSE_FILE" "$TMP_COMPOSE"

# ===================== 【YAML 安全替换】绝对不破坏格式 =====================
sed -i'' -e "/^[[:space:]]*${ORGI_SERVICE}:[[:space:]]*$/{s//  ${NEW_SERVICE_NAME}:/}" \
       -e "/^[[:space:]]*image:/s|image:.*|image: ${IMAGE_NAME}|" \
       -e "/^[[:space:]]*container_name:/s|container_name:.*|container_name: ${CONTAINER_NAME}|" \
       -e "/^[[:space:]]*working_dir:/s|working_dir:.*|working_dir: ${WORKING_DIR}|" \
       -e "/^[[:space:]]*dockerfile:/s|dockerfile:.*|dockerfile: ${DOCKERFILE}|" \
       "$TMP_COMPOSE"

# 启动前校验 service 是否真的改名成功，避免 no such service
if ! grep -qE "^[[:space:]]*${NEW_SERVICE_NAME}:[[:space:]]*$" "$TMP_COMPOSE"; then
    echo -e "\033[31m❌ 未在临时 compose 中找到 service: ${NEW_SERVICE_NAME}\033[0m"
    echo -e "\033[33m可用 service 列表：\033[0m"
    sed -n '/^services:/,/^[^[:space:]]/p' "$TMP_COMPOSE" | grep -E "^[[:space:]]{2}[A-Za-z0-9_-]+:"
    rm -f "$TMP_COMPOSE"
    exit 1
fi

# ===================== 清理旧容器/镜像 =====================
echo -e "\n\033[34m=== 清理同名旧容器 & 旧镜像 ===\033[0m"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rmi -f "$IMAGE_NAME" >/dev/null 2>&1 || true

# ===================== 启动容器（绝对成功） =====================
echo -e "\n\033[32m🚀 启动容器：$NEW_SERVICE_NAME（使用 ${DOCKERFILE}）\033[0m"
if [ "$MODE" = "1" ]; then
    echo -e "\033[34m  → apt: 阿里云 | pip: 清华 | conda: 清华 | Miniconda: bfsu\033[0m"
fi
docker compose -f "$TMP_COMPOSE" up --build -d "$NEW_SERVICE_NAME"

# ===================== 展示结果 & 清理 =====================
echo -e "\n\033[32m🎉 容器启动成功！\033[0m"
docker ps | grep "$CONTAINER_NAME"

rm -f "$TMP_COMPOSE"
echo -e "\n🧹 临时文件已清理，原文件 untouched ✅"
