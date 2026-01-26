#!/bin/bash

# Script thiết lập môi trường và tải dữ liệu cho Mining Massive Dataset Project
# Hỗ trợ cả Linux và Windows (qua Git Bash/WSL)

set -e  # Dừng script nếu có lỗi

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hàm in thông báo
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Hàm hiển thị progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\rProgress: ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %d%%" $percentage
}

# Kiểm tra hệ điều hành
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="Linux";;
        Darwin*)    OS="Mac";;
        CYGWIN*|MINGW*|MSYS*|MINGW32*|MINGW64*)    OS="Windows";;
        *)          OS="Unknown";;
    esac
    print_step "Đã phát hiện hệ điều hành: $OS"
}

# Kiểm tra conda/miniconda đã cài đặt chưa
check_conda() {
    print_step "Đang kiểm tra cài đặt Conda/Miniconda/Anaconda..."
    
    if command -v conda &> /dev/null; then
        CONDA_VERSION=$(conda --version)
        print_success "Đã tìm thấy: $CONDA_VERSION"
        return 0
    fi
    
    # Nếu conda không có trong PATH, tìm trong các đường dẫn phổ biến
    print_warning "Lệnh conda không có trong PATH, đang tìm kiếm..."
    
    # Danh sách các đường dẫn có thể
    POSSIBLE_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/Miniconda3"
        "$HOME/Anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
        "/opt/conda"
    )
    
    for CONDA_PATH in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$CONDA_PATH/bin/conda" ]; then
            print_success "Đã tìm thấy conda tại: $CONDA_PATH"
            export PATH="$CONDA_PATH/bin:$PATH"
            CONDA_VERSION=$(conda --version)
            print_success "Đã khởi tạo: $CONDA_VERSION"
            return 0
        fi
    done
    
    print_error "Không tìm thấy Conda/Miniconda/Anaconda!"
    print_warning "Vui lòng cài đặt Miniconda từ: https://docs.conda.io/en/latest/miniconda.html"
    print_warning "Hoặc đảm bảo conda đã được thêm vào PATH"
    exit 1
}

# Kiểm tra Node.js và npm đã cài đặt chưa
check_nodejs() {
    print_step "Đang kiểm tra cài đặt Node.js và npm..."
    
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_success "Đã tìm thấy: Node.js $NODE_VERSION, npm $NPM_VERSION"
        return 0
    else
        print_warning "Không tìm thấy Node.js hoặc npm!"
        print_warning "Vui lòng cài đặt Node.js từ: https://nodejs.org/"
        print_warning "Bỏ qua bước npm install..."
        return 1
    fi
}

# Cài đặt dependencies từ package.json
npm_install() {
    print_step "Đang cài đặt npm dependencies..."
    
    if [ ! -f "package.json" ]; then
        print_warning "Không tìm thấy file package.json, bỏ qua npm install"
        return 0
    fi
    
    # Kiểm tra xem có dependencies không
    if ! grep -q '"dependencies"' package.json && ! grep -q '"devDependencies"' package.json; then
        print_step "Không có dependencies trong package.json"
        return 0
    fi
    
    print_step "Đang chạy npm install..."
    npm install
    
    if [ $? -eq 0 ]; then
        print_success "Đã cài đặt npm dependencies thành công"
    else
        print_error "Có lỗi khi cài đặt npm dependencies"
        return 1
    fi
}

# Khởi tạo conda cho shell hiện tại
init_conda() {
    print_step "Đang khởi tạo Conda..."
    
    # Tìm conda path
    CONDA_BASE=$(conda info --base 2>/dev/null)
    
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        print_success "Đã khởi tạo Conda"
    else
        print_warning "Không thể tìm thấy conda.sh, tiếp tục với conda có sẵn"
    fi
}

# Cài đặt hoặc cập nhật môi trường từ environment.yml
setup_environment() {
    print_step "Đang thiết lập môi trường conda..."
    
    if [ ! -f "environment.yml" ]; then
        print_error "Không tìm thấy file environment.yml!"
        exit 1
    fi
    
    # Lấy tên môi trường từ environment.yml
    ENV_NAME=$(grep "^name:" environment.yml | awk '{print $2}')
    
    if [ -z "$ENV_NAME" ]; then
        print_error "Không thể đọc tên môi trường từ environment.yml"
        exit 1
    fi
    
    print_step "Tên môi trường: $ENV_NAME"
    
    # Kiểm tra môi trường đã tồn tại chưa
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Môi trường '$ENV_NAME' đã tồn tại"
        read -p "Bạn có muốn cập nhật môi trường? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Đang cập nhật môi trường..."
            conda env update -f environment.yml --prune
            print_success "Đã cập nhật môi trường '$ENV_NAME'"
        else
            print_step "Bỏ qua cập nhật môi trường"
        fi
    else
        print_step "Đang tạo môi trường mới '$ENV_NAME'..."
        echo
        conda env create -f environment.yml
        echo
        print_success "Đã tạo môi trường '$ENV_NAME'"
    fi
}

# Tải file từ Google Drive
download_from_gdrive() {
    local file_id=$1
    local output_path=$2
    
    print_step "Đang tải dữ liệu từ Google Drive..."
    
    # Tạo thư mục nếu chưa có
    mkdir -p "$(dirname "$output_path")"
    
    # URL download trực tiếp từ Google Drive
    local gdrive_url="https://drive.google.com/uc?export=download&id=${file_id}"
    
    # Kiểm tra công cụ download có sẵn
    if command -v wget &> /dev/null; then
        print_step "Sử dụng wget để tải file..."
        wget --progress=bar:force:noscroll -O "$output_path" "$gdrive_url" 2>&1 | \
            grep --line-buffered -oP '\d+%' | \
            while read -r percent; do
                printf "\rDownload progress: %s" "$percent"
            done
        echo
    elif command -v curl &> /dev/null; then
        print_step "Sử dụng curl để tải file..."
        curl -L -o "$output_path" "$gdrive_url" \
            --progress-bar \
            2>&1 | tr '\r' '\n' | tail -n 1
    else
        print_error "Không tìm thấy wget hoặc curl để tải file!"
        print_warning "Vui lòng cài đặt wget hoặc curl"
        exit 1
    fi
    
    # Kiểm tra file đã tải thành công chưa
    if [ -f "$output_path" ]; then
        FILE_SIZE=$(du -h "$output_path" | cut -f1)
        print_success "Đã tải file thành công! Kích thước: $FILE_SIZE"
        
        # Kiểm tra nếu file quá nhỏ (có thể là error page từ Google)
        if [ $(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null) -lt 10000 ]; then
            print_warning "File tải về có vẻ quá nhỏ. Có thể cần xác nhận trên Google Drive."
            print_warning "Đang thử tải lại bằng gdown..."
            if command -v gdown &> /dev/null; then
                gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"
            else
                print_error "Không tìm thấy gdown. Vui lòng kích hoạt môi trường conda trước."
            fi
        fi
    else
        print_error "Không thể tải file!"
        exit 1
    fi
}

# Main script
main() {
    echo "=========================================="
    echo "   Mining Massive Dataset - Setup Script"
    echo "=========================================="
    echo
    
    # 1. Phát hiện hệ điều hành
    detect_os
    echo
    
    # 2. Kiểm tra Conda
    check_conda
    echo
    
    # 3. Kiểm tra Node.js và npm
    NODE_AVAILABLE=false
    if check_nodejs; then
        NODE_AVAILABLE=true
    fi
    echo
    
    # 4. Khởi tạo Conda
    init_conda
    echo
    
    # 5. Thiết lập môi trường
    setup_environment
    echo
    
    # 6. Cài đặt npm dependencies (nếu có Node.js)
    if [ "$NODE_AVAILABLE" = true ]; then
        npm_install
        echo
    fi
    
    # 7. Tải dữ liệu
    print_step "Đang chuẩn bị tải dữ liệu..."
    
    # Trích xuất file ID từ Google Drive link
    # Link: https://drive.google.com/file/d/1VowaeceMECgMEQ-G70-5pSroNG7ggcHB/view
    FILE_ID="1VowaeceMECgMEQ-G70-5pSroNG7ggcHB"
    OUTPUT_PATH="data/raw/accepted_2007_to_2018Q4.csv"
    
    # Kiểm tra file đã tồn tại chưa
    if [ -f "$OUTPUT_PATH" ]; then
        print_warning "File dữ liệu đã tồn tại: $OUTPUT_PATH"
        read -p "Bạn có muốn tải lại? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Bỏ qua tải file"
        else
            download_from_gdrive "$FILE_ID" "$OUTPUT_PATH"
        fi
    else
        download_from_gdrive "$FILE_ID" "$OUTPUT_PATH"
    fi
    
    echo
    echo "=========================================="
    print_success "Hoàn thành thiết lập!"
    echo "=========================================="
    echo
    print_step "Các bước tiếp theo:"
    echo "  1. Kích hoạt môi trường: conda activate MMD"
    echo "  2. Chạy notebook EDA trong thư mục eda/"
    echo
    print_step "Nếu tải dữ liệu thất bại, bạn có thể:"
    echo "  1. Kích hoạt môi trường: conda activate MMD"
    echo "  2. Tải thủ công: gdown $FILE_ID -O $OUTPUT_PATH"
    echo "  3. Hoặc tải trực tiếp từ trình duyệt và đặt vào data/raw/"
    echo
}

# Chạy script
main
