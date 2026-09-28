# Defaults reproduce the documented training settings with portable local paths.
[CmdletBinding()]
param(
    [string]$RepoId = 'BoyuZhao/so101_test',
    [string]$DatasetRoot = '',
    [string]$OutputDir = '',
    [ValidateRange(1, 2147483647)][int]$Steps = 120000,
    [ValidateRange(1, 65536)][int]$BatchSize = 8,
    [ValidateRange(1, 2147483647)][int]$SaveFreq = 10000,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (!$DatasetRoot) { $DatasetRoot = Join-Path $PSScriptRoot '../data/so101_test' }
if (!$OutputDir) { $OutputDir = Join-Path $PSScriptRoot '../outputs/train/act_so101_test_120k' }
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (!$DryRun) {
    if (!(Test-Path -LiteralPath (Join-Path $DatasetRoot 'meta/info.json'))) { throw 'DatasetRoot must contain a complete LeRobot dataset (meta/info.json, data and videos).' }
    if (Test-Path -LiteralPath $OutputDir) { throw 'OutputDir already exists. Choose a new output directory; this script starts a fresh run.' }
}
$cliArgs = @(
    "--dataset.repo_id=$RepoId", "--dataset.root=$DatasetRoot", '--dataset.streaming=false',
    '--policy.type=act', "--output_dir=$OutputDir", '--job_name=act_so101_test_120k',
    '--policy.device=cuda', '--wandb.enable=false', '--policy.push_to_hub=false',
    "--steps=$Steps", "--batch_size=$BatchSize", "--save_freq=$SaveFreq", '--policy.use_amp=true'
)
if ($DryRun) { 'lerobot-train'; $cliArgs; return }
Get-Command lerobot-train -ErrorAction Stop | Out-Null
& lerobot-train @cliArgs
if ($LASTEXITCODE -ne 0) { throw "lerobot-train failed with exit code $LASTEXITCODE" }
