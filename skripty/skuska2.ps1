
$ProcMonTest = "C:\PycharmProjects\Diplomka\skusobny\programy"
$CSVFile = "C:\PycharmProjects\Diplomka\skusobny\procout\notepad.csv"
$ProcMon = "D:\proc\Procmon.exe"
$HandleExe = "D:\proc\handle.exe"
$ProcMonBack = "D:\temp\ProcMonTest.pml"
$FileLocked = $false

Get-ChildItem $ProcMonTest | 
Foreach-Object {
write $_.FullName
write $_.BaseName
}