$ProcMonTest = "C:\PycharmProjects\Diplomka\skusobny\programy\CFF.exe"
$CSVFile = "C:\PycharmProjects\Diplomka\skusobny\procout\notepad.csv"
$ProcMon = "D:\proc\Procmon.exe"
$HandleExe = "D:\proc\handle.exe"
$ProcMonBack = "D:\temp\ProcMonTest.pml"
$FileLocked = $false


# make sure backing file isn't present in case it wasn't deleted on last run
$FileExists = Test-Path $ProcMonBack

if ($FileExists -eq $true){
     do {
        Start-Sleep -seconds 1
        $TestFileLock = & $HandleExe $ProcMonBack /AcceptEula
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
    } while( $FileLocked -eq $true )
    write $TestFileLock[-2]
   Remove-Item $ProcMonBack -force
}
 
& $ProcMon /Quiet /AcceptEula /Minimized /Backingfile $ProcMonBack

do {
    Start-Sleep -seconds 1 # procmon.exe /waitforidle doesn't appear to work well when scripted with PowerShell
    $ProcMonProcess = Get-Process | where {$_.Path -eq $ProcMon}
} while ( $ProcMonProcess.Id -eq $null )

& $ProcMonTest -WindowStyle Hidden
Start-Sleep -seconds 5  # cas ktory sa zaznamenava proces

$ProcMonTestProcess = Get-Process | where {$_.Path -eq $ProcMonTest}
Stop-Process $ProcMonTestProcess.Id
Wait-Process $ProcMonTestProcess.Id

#$ProcMonProcess = Get-Process | where {$_.Path -eq $ProcMon}

& $ProcMon /Terminate
#Start-Process -FilePath $ProcMon -argument "/terminate /accepteula" -Wait -WindowStyle Hidden

# Test for file lock on procmon.exe backing file before exporting
do {
    Start-Sleep -seconds 1
    $TestFileLock = & $HandleExe $ProcMonBack /AcceptEula
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
} while( $FileLocked -eq $true )
 
# Read the procmon.exe backing file and export as CSV
& $ProcMon /openlog $ProcMonBack /SaveAs $CSVFile /SaveApplyFilter

$ProcMonProcess = Get-Process | where {$_.Path -eq $ProcMon}
& $ProcMon /Terminate
 
# Clean up procmon.exe backing file
$FileExists = Test-Path $ProcMonBack

if ($FileExists -eq $true){
     do {
        Start-Sleep -seconds 1
        $TestFileLock = & $HandleExe $ProcMonBack /AcceptEula
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
    } while( $FileLocked -eq $true )
    write $TestFileLock[-2]
   Remove-Item $ProcMonBack -force
}

do {
    Start-Sleep -seconds 1
    $TestFileLock = & $HandleExe $CSVFile /AcceptEula
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
} while( $FileLocked -eq $true )