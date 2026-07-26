# TGFM: Reliability-Routed Low-SNR Modulation Recognition

本仓库是 TGFM 最终实验方法的精简复现代码。核心流程由高信噪比 Teacher、可靠性路由器和低信噪比 Raw-IQ Expert 组成，用于 unknown-SNR 场景下的自动调制识别。

## 目录

```text
.
├── configs/                  # RML2018 主实验配置
├── scripts/                  # 训练、评估与跨数据集实验入口
├── splits/                   # 固定随机种子的数据划分索引
├── src/tgfm/                 # 数据集、模型与通用工具
├── docs/                     # 发布前说明与合规审查
├── pyproject.toml
└── README.md
```

核心文件：

- `src/tgfm/models.py`：Teacher、对齐模块及模型构建函数。
- `src/tgfm/data.py`：RML2018 与 TechReg 数据读取。
- `scripts/train_teacher.py`：Teacher 训练。
- `scripts/train_reliability_router.py`：可靠性路由器训练。
- `scripts/train_raw_low_snr_expert.py`：Raw-IQ 低信噪比专家训练。
- `scripts/evaluate_v8_raw_expert_diagnostics.py`：最终混合模型及分信噪比诊断。
- `scripts/run_rml2016_tgfm128.py`、`scripts/run_techreg_v8_probe.py`：跨数据集验证。

## 环境

建议使用 Python 3.10+ 和支持 CUDA 的 PyTorch 环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 数据准备

原始数据和模型权重未包含在仓库中。默认目录如下：

```text
data/
├── RML2018/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5
├── RML201610a/
│   └── RML2016.10a_dict.pkl
├── RML201610b/
│   └── 2016.10b/*.txt
└── TechReg/
    ├── gentbrugge/
    ├── igent/
    ├── merelbeke/
    ├── rabot/
    ├── reep/
    └── uz/
```

如数据位于其他位置，请修改 `configs/rml2018_tgfm_v7_curriculum.yaml` 中的 `paths`，或为 RML2016 脚本传入 `--data-root`。

## RML2018 主实验

Linux/macOS 可直接运行端到端脚本：

```bash
bash scripts/run_rml2018_pipeline.sh
```

该脚本依次训练 Teacher、可靠性路由器、time-only Raw-IQ Expert，并运行最终诊断。已有 Teacher 或路由器时可复用检查点：

```bash
TEACHER_RUN=runs/teacher_xxx \
ROUTER_RUN=runs/v7_reliability_router_xxx \
bash scripts/run_rml2018_pipeline.sh
```

也可逐步执行：

```bash
python scripts/train_teacher.py \
  --config configs/rml2018_tgfm_v7_curriculum.yaml

python scripts/train_reliability_router.py \
  --config configs/rml2018_tgfm_v7_curriculum.yaml \
  --teacher-run runs/teacher_xxx

python scripts/train_raw_low_snr_expert.py \
  --config configs/rml2018_tgfm_v7_curriculum.yaml \
  --teacher-run runs/teacher_xxx \
  --router-run runs/v7_reliability_router_xxx \
  --branch-mode time_only \
  --seed-override 123 \
  --run-tag time_only_seed123

python scripts/evaluate_v8_raw_expert_diagnostics.py \
  --config configs/rml2018_tgfm_v7_curriculum.yaml \
  --teacher-run runs/teacher_xxx \
  --router-run runs/v7_reliability_router_xxx \
  --expert-run runs/v8_raw_low_snr_expert_time_only_seed123_xxx \
  --output-tag time_only_seed123
```

## 跨数据集验证

RML2016.10a：

```bash
python scripts/run_rml2016_tgfm128.py --dataset 10a
```

TechReg 需要传入 RML2018 的三个检查点目录：

```bash
python scripts/run_techreg_v8_probe.py \
  --mode raw_time \
  --teacher-run runs/teacher_xxx \
  --router-run runs/v7_reliability_router_xxx \
  --expert-run runs/v8_raw_low_snr_expert_time_only_seed123_xxx
```

## 说明

- `runs/`、数据集、缓存和权重已通过 `.gitignore` 排除。
- `splits/rml2018_seed123.npz` 仅包含训练/验证/测试样本索引和随机种子，不包含原始信号。
- 上述 split 文件的 SHA-256 为 `98bd282ba84d629ce0f98ca6d3c168a1aa0eee12c47a3d64037897c2e4076687`。
- 上传前请确认代码权属并选择合适的软件许可证；当前整理包未擅自添加许可证。
- 发布检查见 `docs/ETHICS_REVIEW.md`。
