param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [int]$Seconds = 2
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..").Path
$env:PYTHONPATH = Join-Path $rootDir "src"

python - <<PYTHON
from zerocopy.io.chunker import segment_video
segment_video(r"$InputPath", r"$OutputDir", sec=$Seconds)
PYTHON
