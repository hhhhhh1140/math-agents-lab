# 真实模型小样本测试报告

日期：2026-11-14。模型：SiliconFlow / Qwen/Qwen2.5-7B-Instruct。

## 网络与兼容性修复

- 原容器直连失败原因是 DNS 解析失败；当前 Clash 路径在 TLS 握手时中断。
- 应用容器设置专用 DNS（223.5.5.5、119.29.29.29），API 明确直连并忽略全局代理变量；系统代理保持不变。
- 修复后 /models 认证 HTTP 200，约 0.23 秒，配置模型可用。
- 启用 JSON 模式，解决首轮测试中的非法 JSON 和 Markdown 代码块输出问题。
- 修复启动脚本准备数据时覆盖 dataset 变量、导致结果目录标签错误的问题。首轮 GSM8K 目录已更名，原 manifest 中 output 保留原始路径。
- 容器内回归测试：23 项通过。

## JSON 模式复测

两个数据集各取 2 题，seed=42；每题运行三种模式，共 12 条结果。仅用于验证连接与兼容性，不能代表整体基准准确率。

| 数据集 | 模式 | 正确/样本 | 平均轮数 | 模型调用数 | 输入 token | 输出 token |
|---|---|---|---:|---:|---:|---:|
| gsm8k | direct | 1/2 | 1.0 | 2 | 207 | 12 |
| gsm8k | multi | 1/2 | 3.0 | 26 | 7196 | 1138 |
| gsm8k | single | 2/2 | 3.0 | 6 | 1310 | 227 |
| math500 | direct | 1/2 | 1.0 | 2 | 159 | 33 |
| math500 | multi | 1/2 | 3.0 | 26 | 8188 | 2247 |
| math500 | single | 0/2 | 3.0 | 6 | 1410 | 393 |

复测共使用 22,520 token；含首次诊断实验共 43,377 token。未配置 token 单价，无法据此给出费用。
复测网络/API 异常：0；JSON 解析错误：0。
四次 multi 运行均达到 3 轮上限且 accepted=false；离线评分仍按最后候选答案计分。可运行不等于数学能力可靠，生成代码和验证逻辑仍有错误。
三种模式轮数的定义不同，应同时比较模型调用数和 token。

## 文件

- `results/gsm8k-20261114-180917-8742`：逐题 trace、manifest 与 summary。
- `results/math500-20261114-180917-8754`：逐题 trace、manifest 与 summary。
- `results/gsm8k-initial-20261114-180705-8287`：逐题 trace、manifest 与 summary。
- `results/math500-20261114-180710-8327`：逐题 trace、manifest 与 summary。
