# Setup Guide

This document provides step-by-step instructions for setting up the Mining Massive Dataset project environment.

## Quick Start

### Windows
```powershell
# Chạy PowerShell script
.\setup.ps1
```

### Linux / macOS
```bash
# Chạy bash script
./setup.sh
```

## Prerequisites

- **Conda/Miniconda**: Required for Python environment management
  - Download: https://docs.conda.io/en/latest/miniconda.html
- **Node.js** (Optional): For npm packages
  - Download: https://nodejs.org/
- **Git**: For version control
  - Windows: https://git-scm.com/download/win
  - Linux: `sudo apt-get install git` hoặc `sudo yum install git`
  - macOS: `brew install git`

## What the Setup Script Does

1. ✅ Detects your operating system
2. ✅ Checks for Conda/Miniconda installation
3. ✅ Checks for Node.js and npm (optional)
4. ✅ Creates/updates conda environment from `environment.yml`
5. ✅ Installs npm dependencies (if available)
6. ✅ Downloads dataset from Google Drive
7. ✅ Sets up project structure

## Manual Setup (If Scripts Fail)

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
```

### 2. Activate Environment
```bash
conda activate MMD
```

### 3. Install npm packages (Optional)
```bash
npm install
```

### 4. Download Dataset
Có 3 cách tải dataset:

#### Option A: Using gdown (Recommended)
```bash
conda activate MMD
gdown 1VowaeceMECgMEQ-G70-5pSroNG7ggcHB -O data/raw/accepted_2007_to_2018Q4.csv
```

#### Option B: Using wget
```bash
wget https://drive.google.com/uc?export=download&id=1VowaeceMECgMEQ-G70-5pSroNG7ggcHB -O data/raw/accepted_2007_to_2018Q4.csv
```

#### Option C: Manual Download
1. Truy cập: https://drive.google.com/file/d/1VowaeceMECgMEQ-G70-5pSroNG7ggcHB/view
2. Tải file về máy
3. Đặt file vào thư mục `data/raw/accepted_2007_to_2018Q4.csv`

## Troubleshooting

### Windows Issues

#### PowerShell Execution Policy
Nếu gặp lỗi execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### WSL Error
Nếu gặp lỗi WSL khi chạy `bash setup.sh`:
- **Giải pháp 1**: Sử dụng PowerShell script thay vì bash
  ```powershell
  .\setup.ps1
  ```
- **Giải pháp 2**: Cài đặt Git Bash và chạy từ Git Bash terminal
- **Giải pháp 3**: Cài đặt WSL đúng cách từ Microsoft Store

### Conda Not Found
```bash
# Thêm conda vào PATH (Windows)
# Tìm đường dẫn cài đặt conda (thường là C:\Users\<username>\miniconda3)
# Thêm vào System Environment Variables

# Linux/macOS: Thêm vào ~/.bashrc hoặc ~/.zshrc
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc  # hoặc source ~/.zshrc
```

### Large File Download Issues
Nếu file từ Google Drive quá lớn và không tải được:
1. Kích hoạt môi trường: `conda activate MMD`
2. Sử dụng gdown: `gdown 1VowaeceMECgMEQ-G70-5pSroNG7ggcHB -O data/raw/accepted_2007_to_2018Q4.csv`
3. Hoặc tải thủ công từ browser

## Verify Installation

```bash
# Kích hoạt môi trường
conda activate MMD

# Kiểm tra Python
python --version

# Kiểm tra packages
conda list

# Kiểm tra dataset
ls -lh data/raw/accepted_2007_to_2018Q4.csv
```

## Next Steps

Sau khi setup thành công:

1. **Kích hoạt môi trường**
   ```bash
   conda activate MMD
   ```

2. **Khám phá EDA notebooks**
   ```bash
   cd eda
   jupyter notebook
   ```

3. **Xem tài liệu dự án**
   - Requirements: `docs/ai/requirements/README.md`
   - Design: `docs/ai/design/README.md`
   - Planning: `docs/ai/planning/README.md`