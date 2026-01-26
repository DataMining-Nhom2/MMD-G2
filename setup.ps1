# Script setup environment and download data for Mining Massive Dataset Project
# PowerShell script for Windows

# Stop script on error
$ErrorActionPreference = "Stop"

# Color output functions
function Print-Step {
    param([string]$Message)
    Write-Host "==> " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Print-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

# Progress bar display function
function Show-Progress {
    param(
        [int]$Current,
        [int]$Total
    )
    $percentage = [math]::Round(($Current / $Total) * 100)
    $filled = [math]::Round(($Current / $Total) * 50)
    $empty = 50 - $filled
    
    Write-Host "`rProgress: [" -NoNewline
    Write-Host ("=" * $filled) -NoNewline -ForegroundColor Green
    Write-Host ("." * $empty) -NoNewline -ForegroundColor Gray
    Write-Host "] $percentage%" -NoNewline
}

# Check conda/miniconda installation
function Check-Conda {
    Print-Step "Checking Conda/Miniconda/Anaconda installation..."
    
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $condaVersion = conda --version 2>$null
        Print-Success "Found: $condaVersion"
        return $true
    }
    
    # If conda not in PATH, try to find it in common installation locations
    Print-Warning "conda command not found in PATH, searching common locations..."
    
    $possiblePaths = @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\Miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\Anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "$env:LOCALAPPDATA\Miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\Anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\Miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\Anaconda3"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path "$path\Scripts\conda.exe") {
            Print-Success "Found conda at: $path"
            $env:Path = "$path;$path\Scripts;$path\Library\bin;" + $env:Path
            
            # Try to initialize conda
            try {
                (& "$path\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
                $condaVersion = conda --version 2>$null
                Print-Success "Initialized: $condaVersion"
                return $true
            } catch {
                Print-Warning "Found conda but failed to initialize: $_"
            }
        }
    }
    
    Print-Error "Conda/Miniconda/Anaconda not found!"
    Print-Warning "Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    Print-Warning "Or make sure conda is added to PATH"
    Print-Warning "After installation, restart PowerShell"
    exit 1
}

# Check Node.js and npm installation
function Check-NodeJS {
    Print-Step "Checking Node.js and npm installation..."
    
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    
    if ($nodeCmd -and $npmCmd) {
        $nodeVersion = node --version
        $npmVersion = npm --version
        Print-Success "Found: Node.js $nodeVersion, npm $npmVersion"
        return $true
    } else {
        Print-Warning "Node.js or npm not found!"
        Print-Warning "Please install Node.js from: https://nodejs.org/"
        Print-Warning "Skipping npm install..."
        return $false
    }
}

# Install dependencies from package.json
function Install-NpmPackages {
    Print-Step "Installing npm dependencies..."
    
    if (-not (Test-Path "package.json")) {
        Print-Warning "package.json not found, skipping npm install"
        return
    }
    
    # Check if dependencies exist
    $packageJson = Get-Content "package.json" | ConvertFrom-Json
    if (-not $packageJson.dependencies -and -not $packageJson.devDependencies) {
        Print-Step "No dependencies in package.json"
        return
    }
    
    Print-Step "Running npm install..."
    try {
        npm install
        Print-Success "npm dependencies installed successfully"
    } catch {
        Print-Error "Error installing npm dependencies: $_"
    }
}

# Initialize conda for PowerShell
function Initialize-Conda {
    Print-Step "Initializing Conda..."
    
    try {
        # Get conda base path
        $condaBase = (conda info --base) 2>$null
        
        if ($condaBase -and (Test-Path "$condaBase\Scripts\conda.exe")) {
            # Initialize conda for current PowerShell session
            (& "$condaBase\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
            Print-Success "Conda initialized"
        } else {
            Print-Warning "Cannot initialize Conda, continuing with available conda"
        }
    } catch {
        Print-Warning "Cannot initialize Conda: $_"
    }
}

# Create or update environment from environment.yml
function Setup-Environment {
    Print-Step "Setting up conda environment..."
    
    if (-not (Test-Path "environment.yml")) {
        Print-Error "environment.yml not found!"
        exit 1
    }
    
    # Get environment name from environment.yml
    $envName = (Get-Content "environment.yml" | Select-String "^name:").ToString().Split(":")[1].Trim()
    
    if (-not $envName) {
        Print-Error "Cannot read environment name from environment.yml"
        exit 1
    }
    
    Print-Step "Environment name: $envName"
    
    # Check if environment already exists
    $envExists = conda env list | Select-String "^$envName\s"
    
    if ($envExists) {
        Print-Warning "Environment '$envName' already exists"
        $response = Read-Host "Do you want to update the environment? (y/n)"
        
        if ($response -match "^[Yy]$") {
            Print-Step "Updating environment..."
            conda env update -f environment.yml --prune
            Print-Success "Environment '$envName' updated"
        } else {
            Print-Step "Skipping environment update"
        }
    } else {
        Print-Step "Creating new environment '$envName'..."
        Write-Host ""
        conda env create -f environment.yml
        Write-Host ""
        Print-Success "Environment '$envName' created"
    }
}

# Download file from Google Drive
function Download-FromGDrive {
    param(
        [string]$FileId,
        [string]$OutputPath
    )
    
    Print-Step "Downloading data from Google Drive..."
    
    # Create directory if not exists
    $outputDir = Split-Path -Parent $OutputPath
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Direct download URL from Google Drive
    $gdriveUrl = "https://drive.google.com/uc?export=download&id=$FileId"
    
    try {
        Print-Step "Downloading file with Invoke-WebRequest..."
        
        # Download file with progress bar
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $gdriveUrl -OutFile $OutputPath -UseBasicParsing
        $ProgressPreference = 'Continue'
        
        if (Test-Path $OutputPath) {
            $fileSize = (Get-Item $OutputPath).Length
            $fileSizeFormatted = "{0:N2} MB" -f ($fileSize / 1MB)
            Print-Success "File downloaded successfully! Size: $fileSizeFormatted"
            
            # Check if file is too small (might be error page from Google)
            if ($fileSize -lt 10000) {
                Print-Warning "Downloaded file seems too small. May need confirmation on Google Drive."
                Print-Warning "Trying to download again with gdown..."
                
                $gdownCmd = Get-Command gdown -ErrorAction SilentlyContinue
                if ($gdownCmd) {
                    gdown "https://drive.google.com/uc?id=$FileId" -O $OutputPath
                } else {
                    Print-Error "gdown not found. Please activate conda environment first."
                    Print-Warning "Run: conda activate MMD"
                    Print-Warning "Then: gdown $FileId -O $OutputPath"
                }
            }
        } else {
            Print-Error "Cannot download file!"
            exit 1
        }
    } catch {
        Print-Error "Error downloading file: $_"
        Print-Warning "Trying alternative method..."
        
        # Try curl if available
        $curlCmd = Get-Command curl -ErrorAction SilentlyContinue
        if ($curlCmd) {
            Print-Step "Using curl to download file..."
            curl -L -o $OutputPath $gdriveUrl
        } else {
            Print-Error "Cannot download file. Please download manually."
            exit 1
        }
    }
}

# Main script
function Main {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "   Mining Massive Dataset - Setup Script" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # 1. Check Conda
    Check-Conda
    Write-Host ""
    
    # 2. Check Node.js and npm
    $nodeAvailable = Check-NodeJS
    Write-Host ""
    
    # 3. Initialize Conda
    Initialize-Conda
    Write-Host ""
    
    # 4. Setup environment
    Setup-Environment
    Write-Host ""
    
    # 5. Install npm dependencies (if Node.js available)
    if ($nodeAvailable) {
        Install-NpmPackages
        Write-Host ""
    }
    
    # 6. Download data
    Print-Step "Preparing to download data..."
    
    # Extract file ID from Google Drive link
    $fileId = "1VowaeceMECgMEQ-G70-5pSroNG7ggcHB"
    $outputPath = "data\raw\accepted_2007_to_2018Q4.csv"
    
    # Check if file already exists
    if (Test-Path $outputPath) {
        Print-Warning "Data file already exists: $outputPath"
        $response = Read-Host "Do you want to download again? (y/n)"
        
        if ($response -match "^[Yy]$") {
            Download-FromGDrive -FileId $fileId -OutputPath $outputPath
        } else {
            Print-Step "Skipping file download"
        }
    } else {
        Download-FromGDrive -FileId $fileId -OutputPath $outputPath
    }
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Print-Success "Setup completed!"
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Print-Step "Next steps:"
    Write-Host "  1. Activate environment: conda activate MMD"
    Write-Host "  2. Run EDA notebooks in eda/ directory"
    Write-Host ""
    Print-Step "If data download failed, you can:"
    Write-Host "  1. Activate environment: conda activate MMD"
    Write-Host "  2. Manual download: gdown $fileId -O $outputPath"
    Write-Host "  3. Or download from browser and place in data\raw\"
    Write-Host ""
}

# Run script
Main
