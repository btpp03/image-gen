# image-gen

**免 API key 的 AI 生图/改图**，跑在 GitHub Actions 上，用机场节点轮换 IP 来续 HuggingFace ZeroGPU 的免费额度。

当前后端：`Qwen-Image-2.1`（`scripts/qwen_image.py`）。

## 为什么需要代理池

HF 的 ZeroGPU 免费额度是**按调用方 IP 计**的。同一个 IP 用完了就报配额错，换个出口 IP 又是一份额度。
`scripts/nodes_to_singbox.py` 把你机场订阅里的每个节点变成一个本地 HTTP 代理端口，`qwen_image.py` 在
命中配额错误时自动换下一个 IP 重试 —— 一个订阅 ≈ 十几次免费出图。

## 用法

Actions → **qwen-image** → Run workflow：

- `prompt`：图片描述（英文更稳）
- `route`：`pool`（默认，走代理池）/ `direct`（直连）
- `ratio` / `resolution` / `steps` / `seed`：透传给 Space
- `image_urls`：可选，参考图直链（逗号分隔，最多 10 张）→ 改图模式

结果在 run 的 **Artifacts** 里下载（`out/*.png` + `result.txt`）。
run summary 里会列出本次存活的出口 IP。

## 密钥（Settings → Secrets and variables → Actions）

二选一，**订阅优先**：

| Secret | 说明 |
|---|---|
| `JKUN_SUB_URL` | 机场订阅地址（每次跑拉最新节点，推荐） |
| `JKUN_SUB_UA` | 拉订阅的 UA，默认 `v2rayNG/1.9.16`；有些机场认 UA |
| `JKUN_NODES_B64` | 节点分享链接列表的 base64 快照（`base64 -w0 nodes.txt`） |
| `HF_TOKEN` | 可选，有 HF 账号的 token 时额度更大 |

两个都没配 → workflow 直接报错退出（不会静默降级成直连）。

## 本地/手动跑

```bash
# 1) 节点 -> sing-box 配置 + 代理端口列表
python3 scripts/nodes_to_singbox.py nodes.txt --out-config sbx.json --out-pool raw_pool.txt
sing-box check -c sbx.json
sing-box run -c sbx.json &

# 2) 剔除死节点
: > pool.txt
while read -r p; do curl -s -x "$p" --max-time 20 https://api.ipify.org && echo "$p" >> pool.txt; done < raw_pool.txt

# 3) 出图（配额报错会自动换 IP）
python3 scripts/qwen_image.py "a red fox sleeping in autumn leaves" \
  --proxy-pool pool.txt --proxy-policy round-robin --out-dir out
```

`qwen_image.py` 只用标准库；只有 socks 代理才需要 `pip install pysocks`。

## 已知边界

- **节点存活率**：实测一批 13 个节点里 10 个能通、10 个出口 IP 互不相同（同一台机器上的多个"线路"其实是同一个出口，脚本会按出站参数去重）。死节点由 workflow 探活剔除。
- **IPv6-only 节点会被跳过**：GitHub runner 没有 IPv6 出口（`ptxlv6-*` 这类会被自动丢弃）。
- **机场出口是共享机房 IP**：可能已被别人刷掉额度，也可能被 HF 风控；所以池子越大越稳。
- **配额按 IP 计是实测推断**：第一次实跑（配额报错后能否换 IP 继续出图）才是终判。
- 公开仓库的 Actions 分钟数免费无限；**别把节点/订阅写进仓库文件**，只放 Secrets。
