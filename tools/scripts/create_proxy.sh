# create_proxy.sh - 网络代理设置脚本
# 用法:
#   source create_proxy.sh        # 默认启用代理
#   source create_proxy.sh on     # 启用代理
#   source create_proxy.sh off    # 关闭代理
#   source create_proxy.sh status # 查看当前代理状态

# 检测脚本是否通过 source 加载。
is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

# 默认代理配置
DEFAULT_PROXY_IP="127.0.0.1"
DEFAULT_PROXY_PORT="7890"

# 代理地址
PROXY_HTTP="http://${DEFAULT_PROXY_IP}:${DEFAULT_PROXY_PORT}"
PROXY_SOCKS="socks5://${DEFAULT_PROXY_IP}:${DEFAULT_PROXY_PORT}"

# 启用代理
enable_proxy() {
  export http_proxy="$PROXY_HTTP"
  export https_proxy="$PROXY_HTTP"
  export all_proxy="$PROXY_SOCKS"
  export HTTP_PROXY="$PROXY_HTTP"
  export HTTPS_PROXY="$PROXY_HTTP"
  export ALL_PROXY="$PROXY_SOCKS"
  echo "✅ 代理已启用：$PROXY_HTTP"
}

# 关闭代理
disable_proxy() {
  unset http_proxy https_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  echo "❌ 代理已关闭"
}

# 显示状态
show_status() {
  echo "当前代理状态："
  echo "http_proxy  = $http_proxy"
  echo "https_proxy = $https_proxy"
  echo "all_proxy   = $all_proxy"
}

register_aliases() {
  local script_abs_path
  script_abs_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  alias proxyon="source ${script_abs_path} on"
  alias proxyoff="source ${script_abs_path} off"
  alias proxystatus="source ${script_abs_path} status"
}

if ! is_sourced; then
  echo "⚠️ 请使用 source 加载脚本，否则代理变量不会保留在当前终端。"
  echo "   正确示例: source third_party_finetune/tools/scripts/create_proxy.sh on"
fi

# 主逻辑
case "$1" in
  ""|on)
    enable_proxy
    ;;
  off)
    disable_proxy
    ;;
  status)
    show_status
    ;;
  *)
    echo "用法：source create_proxy.sh [on|off|status]"
    ;;
esac

if is_sourced; then
  register_aliases
fi
