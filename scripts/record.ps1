# See docs/setup.md before connecting hardware. -DryRun only prints arguments.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$Task,
    [Parameter(Mandatory)][ValidateSet('true', 'false')][string]$UseDegrees,
    [string]$RepoId = 'BoyuZhao/so101_test',
    [string]$DatasetRoot = '',
    [string]$FollowerPort = 'COM7', [string]$FollowerId = 'myfollower01',
    [string]$LeaderPort = 'COM8', [string]$LeaderId = 'myleader01',
    [int]$WristCamera = 0, [int]$FrontCamera = 1,
    [ValidateRange(1, 100000)][int]$Episodes = 40,
    [ValidateRange(1, 3600)][int]$EpisodeSeconds = 60,
    [ValidateRange(1, 3600)][int]$ResetSeconds = 60,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (!$DatasetRoot) { $DatasetRoot = Join-Path $PSScriptRoot '../data/so101_test' }
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
if ((Test-Path -LiteralPath $DatasetRoot) -and !$DryRun) {
    throw 'DatasetRoot already exists. Choose a new directory; this script does not overwrite or resume data.'
}
# YAML avoids embedded quotation marks being stripped by Windows PowerShell 5.1.
$cameras = "{wrist: {type: opencv, index_or_path: $WristCamera, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: $FrontCamera, width: 640, height: 480, fps: 30}}"
$cliArgs = @(
    '--robot.type=so101_follower', "--robot.port=$FollowerPort", "--robot.id=$FollowerId",
    "--robot.use_degrees=$UseDegrees", "--robot.cameras=$cameras",
    '--teleop.type=so101_leader', "--teleop.port=$LeaderPort", "--teleop.id=$LeaderId",
    "--teleop.use_degrees=$UseDegrees", "--dataset.repo_id=$RepoId", "--dataset.root=$DatasetRoot",
    "--dataset.single_task=$Task", '--dataset.fps=30', "--dataset.num_episodes=$Episodes",
    "--dataset.episode_time_s=$EpisodeSeconds", "--dataset.reset_time_s=$ResetSeconds",
    '--dataset.vcodec=h264', '--dataset.push_to_hub=false', '--display_data=false', '--play_sounds=false'
)
if ($DryRun) { 'lerobot-record'; $cliArgs; return }
Get-Command lerobot-record -ErrorAction Stop | Out-Null
& lerobot-record @cliArgs
if ($LASTEXITCODE -ne 0) { throw "lerobot-record failed with exit code $LASTEXITCODE" }
