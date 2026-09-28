# This is real robot motion, using the pinned fork's lerobot-rollout CLI.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$Task,
    [Parameter(Mandatory)][ValidateSet('true', 'false')][string]$UseDegrees,
    [string]$PolicyPath = '',
    [string]$FollowerPort = 'COM7', [string]$FollowerId = 'myfollower01',
    [int]$WristCamera = 0, [int]$FrontCamera = 1,
    [ValidateRange(1, 3600)][int]$DurationSeconds = 30,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (!$PolicyPath) { $PolicyPath = Join-Path $PSScriptRoot '../outputs/train/act_so101_test_120k/checkpoints/last/pretrained_model' }
$PolicyPath = [IO.Path]::GetFullPath($PolicyPath)
if (!$DryRun -and !(Test-Path -LiteralPath (Join-Path $PolicyPath 'config.json'))) {
    throw 'PolicyPath must point to a complete pretrained_model directory, not a single weights file.'
}
$cameras = "{wrist: {type: opencv, index_or_path: $WristCamera, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: $FrontCamera, width: 640, height: 480, fps: 30}}"
$cliArgs = @(
    '--strategy.type=base', '--inference.type=sync', "--policy.path=$PolicyPath",
    '--robot.type=so101_follower', "--robot.port=$FollowerPort", "--robot.id=$FollowerId",
    "--robot.use_degrees=$UseDegrees", "--robot.cameras=$cameras",
    '--device=cuda', '--fps=30', "--duration=$DurationSeconds", "--task=$Task",
    '--return_to_initial_position=false', '--display_data=false', '--play_sounds=false'
)
if ($DryRun) { 'lerobot-rollout'; $cliArgs; return }
Get-Command lerobot-rollout -ErrorAction Stop | Out-Null
& lerobot-rollout @cliArgs
if ($LASTEXITCODE -ne 0) { throw "lerobot-rollout failed with exit code $LASTEXITCODE" }
