# AGENTS.md

## 范围
- 允许算法：**DDPG / PPO / TD3**（任选其一）。
- 只运行 Walker2d 的 **MLP** 部分。
- **完全忽略 LSTM**。
- 任务完成后，**必须记录到 `status.md` 和 `result.md`**。

## 需要运行的内容（仅 MLP）
1) 训练（官方长跑，随机摩擦）— 从允许算法中选择一个：
   ```powershell
   # 官方要求训练（>=1e6 步，随机摩擦）
   .\.venv\Scripts\python.exe src\train.py --critic-arch mlp --seed 42 --total-steps 1000000 --random-friction --visualize-after --record-video
   ```
2) 快速演示（可选短跑，不替代正式实验）：
   ```powershell
   # 短跑演示：快速看到前进（无随机摩擦）
   # 注意：不替代正式实验
   .\.venv\Scripts\python.exe src\train.py --critic-arch mlp --seed 42 --total-steps 200000 --start-steps 1000 --noise-sigma 0.05 --visualize-after --record-video
   ```
3) 超短演示（可选，20k 步内，最快视觉检查）：
   ```powershell
   # 超短演示（<=20k 步），用于最快视觉检查
   .\.venv\Scripts\python.exe src\train.py --critic-arch mlp --seed 42 --total-steps 20000 --start-steps 1000 --noise-sigma 0.05 --actor-lr 0.001 --critic-lr 0.001 --visualize-after --record-video
   ```
4) 评估：
   ```powershell
   # 评估训练好的模型（将 ddpg_mlp_xxx 替换为真实 run_id）
   .\.venv\Scripts\python.exe src\eval.py --model checkpoints\ddpg_mlp_xxx --episodes 20 --seed 123
   ```
5) 绘图：
   ```powershell
   # 绘制奖励曲线（将 ddpg_mlp_xxx 替换为真实 run_id）
   .\.venv\Scripts\python.exe src\plot.py --logdir logs\ddpg_mlp_xxx
   ```

## 可选：TD3（保留 DDPG，TD3 作为额外对比）
```powershell
# TD3 训练（MLP，随机摩擦）- 可选额外实验
.\.venv\Scripts\python.exe src\train.py --algo td3 --critic-arch mlp --seed 42 --total-steps 1000000 --random-friction --visualize-after --record-video
```

## 可选：PPO（仅 MLP）
```powershell
# PPO 训练（MLP，随机摩擦）- 可选额外实验
.\.venv\Scripts\python.exe src\train.py --algo ppo --critic-arch mlp --seed 42 --total-steps 1000000 --random-friction --visualize-after --record-video
```

## 记录要求
- `status.md`：记录已完成的任务（内容、时间、位置），并注明**未完成项**。
- `result.md`：记录运行输出（奖励统计、日志/模型/图像路径）。
 - 例外：当用户明确要求“开始写报告后不再记录”，则暂停更新 `status.md` 与 `result.md`，直到用户撤销该要求。

## 备注
- 严格遵守课程要求（Walker2d-v4、随机摩擦、MLP 对比等）。
- 不新增任何 LSTM 代码或实验。
- 训练必须与评估/绘图/视频保存作为一个完整流程一起完成。
- **所有回答与报告撰写必须使用中文**。
- **PPT 文字版报告（report.md）使用 Markdown 语法，并直接插入图片（`![]()`）显示。**
- **PPT 图片下方的“备注”以“演讲稿”形式撰写，需写明“演讲稿：”并比普通说明更详细。**
