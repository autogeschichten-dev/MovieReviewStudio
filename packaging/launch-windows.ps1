param([string] $SmokeMarker)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root '.windows-runtime'
$python = Join-Path $runtime 'python\python.exe'
$pythonWindowed = Join-Path $runtime 'python\pythonw.exe'
$media = Join-Path $runtime 'ffmpeg\bin'

function Install-Archive($Name, $Url, $Sha256, $Subfolder, $Destination) {
    $temp = Join-Path $runtime ([guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        $archive = Join-Path $temp 'download.zip'
        Write-Host "Downloading $Name..."
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
            throw "$Name failed its SHA-256 integrity check."
        }
        $extracted = Join-Path $temp 'extracted'
        Expand-Archive -LiteralPath $archive -DestinationPath $extracted
        $source = Join-Path $extracted $Subfolder
        if (!(Test-Path -LiteralPath $Destination)) {
            Move-Item -LiteralPath $source -Destination $Destination
        } else {
            Get-ChildItem -LiteralPath $source -Force |
                Copy-Item -Destination $Destination -Recurse -Force
        }
    } finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}

function Invoke-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE. See the preceding output."
    }
}

try {
    if (![Environment]::Is64BitOperatingSystem) {
        throw 'Movie Review Studio requires 64-bit Windows.'
    }
    Set-Location -LiteralPath $root

    # Enable TLS 1.2 for downloads in Windows PowerShell 5.1.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $lock = [IO.File]::Open((Join-Path $runtime 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    try {
        if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $pythonWindowed)) {
            Install-Archive 'Python 3.12 x64' `
                'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg' `
                '0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8' `
                'tools' (Join-Path $runtime 'python')
        }
        if (!(Test-Path -LiteralPath (Join-Path $media 'ffmpeg.exe')) -or
            !(Test-Path -LiteralPath (Join-Path $media 'ffprobe.exe'))) {
            Install-Archive 'FFmpeg and FFprobe' `
                'https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-essentials_build.zip' `
                '04861D3339C5EBE38B56C19A15CF2C0CC97F5DE4FA8910E4D47E5E6404E4A2D4' `
                'ffmpeg-7.1.1-essentials_build' (Join-Path $runtime 'ffmpeg')
        }

        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONHOME = ''
        $env:PYTHONPATH = ''
        $env:PATH = "$media;$env:PATH"
        $pip = Join-Path $runtime 'python\Lib\site-packages\pip\__init__.py'
        if (!(Test-Path -LiteralPath $pip)) {
            Invoke-Python -m ensurepip --default-pip
        }
        Invoke-Python -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: --no-warn-script-location -r requirements.txt
        Invoke-Python -m pip --isolated check
    } finally {
        $lock.Dispose()
    }

    if ($SmokeMarker) {
        $env:QT_QPA_PLATFORM = 'offscreen'
        $env:PYTHONPATH = $root
        Invoke-Python (Join-Path $PSScriptRoot 'windows_entry.py') --startup-smoke-test $SmokeMarker
    } else {
        Start-Process -FilePath $pythonWindowed -ArgumentList '-m', 'movie_review_studio' -WorkingDirectory $root
    }
} catch {
    Write-Host "Movie Review Studio setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Extract the whole ZIP to a writable folder and check your Internet connection before trying again.'
    exit 1
}
