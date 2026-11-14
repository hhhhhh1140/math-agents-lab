# 本机部署记录

日期：2026-11-14（Asia/Shanghai）。

- Docker Engine 29.5.2，Docker CLI 29.9.3，Compose 5.7.1。
- Colima 0.10.3，Apple Silicon 原生虚拟机，2 CPU / 4 GB 内存 / 30 GB 数据盘（按需占用）。
- 应用镜像：`math-agents-lab:local`。
- 代码沙箱镜像：`math-agent-python:local`，Python 3.11 + SymPy 1.14.0。
- `./lab test`：21 项通过，无跳过。
- GSM8K：`data/gsm8k.jsonl`，1,319 道测试题。
- MATH-500：`data/math500.jsonl`，500 道测试题。
- 本机原有 Java 21 已恢复至安装前版本 21.0.13.0.3。

已配置并测试 SiliconFlow 的 Qwen/Qwen2.5-7B-Instruct。应用容器专用 DNS + API 明确直连，启用 JSON 模式。最新扩大实验为两个数据集各 20 道新题、160 条对照记录；容器回归测试 28 项通过。详见 `VALIDATION_REPORT.md`；尚未运行全量评测。

默认使用优化版多 Agent。冻结镜像 `math-agents-lab:baseline-v1` 与 `math-agents-lab:optimized-v2` 保留，样本题号、源代码哈希和镜像标识记录在 `experiments/validation20.json`。可用 `./lab run gsm8k 20 baseline` 切换原流程。

## 用户接下来操作

编辑项目根目录 `.env`，填写 `MATH_API_KEY`、`MATH_BASE_URL`、`MATH_MODEL`。无需向聊天发送密钥。

然后在项目目录运行：

```bash
./lab run gsm8k 20
./lab run math500 20
```

三种方案自动对照，结果保存至 `results/` 的新建时间戳目录。token 单价可选；不填写则费用显示为空。详细参数、全量实验和重新构建方法见 README。
