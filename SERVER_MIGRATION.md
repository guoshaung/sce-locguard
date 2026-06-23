# Stage 9 Server Migration Notes

## 1. Server Target

- GPU: RTX 4090 / RTX 4090D 24GB
- OS: Ubuntu 20.04 or 22.04
- Recommended disk: at least 500GB, preferably 1TB
- RAM: at least 32GB, preferably 64GB

## 2. Local Project

- Local root: `D:\pycharm\watermark_exps`
- GitHub repo: `https://github.com/guoshaung/sce-locguard`
- Dataset zip: `D:\pycharm\watermark_exps\dataset.zip`

The dataset zip is for server upload only. It must not be committed to git.

## 3. Server Project Root

Use:

```bash
/data/watermark_exps
```

## 4. Server Clone Command

```bash
mkdir -p /data/watermark_exps
cd /data/watermark_exps
git clone https://github.com/guoshaung/sce-locguard.git .
```

## 5. Dataset Upload Command Template

From Windows PowerShell:

```powershell
scp D:\pycharm\watermark_exps\dataset.zip USERNAME@SERVER_IP:/data/watermark_exps/
```

If a custom SSH port is needed:

```powershell
scp -P SSH_PORT D:\pycharm\watermark_exps\dataset.zip USERNAME@SERVER_IP:/data/watermark_exps/
```

Do not put passwords in scripts or git history.

## 6. Server Unzip Command

```bash
cd /data/watermark_exps
unzip dataset.zip
```

Expected dataset layout after unzip:

```text
/data/watermark_exps/dataset/valAGE-Set
/data/watermark_exps/dataset/valAGE-Set-Mask
```

## 7. Server Branch Setup

After server clone:

```bash
git checkout -b stage9-server
```

Do not create `stage9-server` locally now unless needed. The server branch will be created after server-side clone.

## 8. Conda Environment Setup

```bash
conda create -n sce-locguard python=3.10 -y
conda activate sce-locguard
```

Check CUDA:

```bash
nvidia-smi
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check PyTorch GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 9. Stage 9 First Run Order

Do not run 1000 samples immediately.

Run order:

1. 5-sample smoke test
2. 50-sample reproduction of Stage 8F
3. 200-sample holdout
4. 1000-sample full evaluation

## 10. Path Policy

All Stage 9 scripts must use command-line path arguments. Do not hard-code Windows `D:/` paths.

Expected server paths:

```text
--project_root /data/watermark_exps
--image_root /data/watermark_exps/dataset/valAGE-Set
--mask_root /data/watermark_exps/dataset/valAGE-Set-Mask
--output_root /data/watermark_exps/dfg_locguard/outputs/stage9_full_eval
--sample_list /data/watermark_exps/dataset_manifest/valage_200_list.txt
```

