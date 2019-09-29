# pouzil som kod z tade: http://nomanualrequired.blogspot.com/
#  a inspiroval som sa tymto: https://trwagner1.wordpress.com/2010/03/25/using-powershell-with-process-monitor/ 
#
	function convert-FilterToObj
	{	   
	   $FilterObj = @()  #Init blank array
	   $Filter | % {
		  #Use builtin split method to convert comma seperated string into array
		  $_ = $_.split(',')
		  #Checking our array for exactly four objects
		  if(($_ | measure).count -ne 4)
		  	{return "Error: Filter is not in correct format."}
		  else
		  {
		    #Convert filter into an object, remove spaces from Column/Relation
		    $CurrentFilter = New-Object system.Object
		    $CurrentFilter | Add-Member noteproperty -Name "Column" -Value $_[0].replace(' ','')
			$CurrentFilter | Add-Member noteproperty -Name "Relation" -Value $_[1].replace(' ','')
			$CurrentFilter | Add-Member noteproperty -Name "Value" -Value $_[2]
			$CurrentFilter | Add-Member noteproperty -Name "Action" -Value $_[3]
			#Add current filter to output object
			$FilterObj += $CurrentFilter
	      }
	   }
	return $FilterObj
	}

	function convert-LargeValues
	{
		param($Value)
		if($Value.Length -gt 2)
		{
			$FirstByte =  [string]::Join("",$Value[1..2])
			$SecondByte = $Value[0]
		}
		else
		{
			$FirstByte = $Value
			$SecondByte = 0
		}
		return "0x$FirstByte","0x$SecondByte"
	}

	function write-ProcmonFilterValue
	{
	    #Start of filter is 1 declare type as an array of bytes
	    [Byte[]]$FilterRegkey = "0x1"
	    #Followed by number of filters
	    $NumFilters = [convert]::tostring((($FilterObj | measure).count),"16")
	    #Multiple registry keys can overflow when using largeValues
		#A function was created to split these large values into two bytes
		$FilterRegKey += (convert-LargeValues -value $NumFilters)
		#Two padding bytes
		$FilterRegkey += "0x0","0x0"
	    #Header is written, build filters from friendly strings
	    $FilterObj | %	{
		  #Check for syntax errors
		  if($FilterRegkey -match "Error")
		   	  {return} 
		  #First write column code, and 9c divider
	      #switch($_.Column)
	      #{
	      #  "ProcessName"      {$FilterRegKey += "0x75","0x9c"}
	      #  "PID"              {$FilterRegKey += "0x76","0x9c"}
	      #  "Result"           {$FilterRegKey += "0x78","0x9c"}
	      #  "Detail"           {$FilterRegkey += "0x79","0x9c"}
	      #  "Duration"         {$FilterRegKey += "0x8d","0x9c"}
	      #  "ImagePath"        {$FilterRegKey += "0x84","0x9c"}
	   	  #  "RelativeTime"     {$FilterRegKey += "0x8c","0x9c"}
	      #  "CommandLine"      {$FilterRegKey += "0x82","0x9c"}
			#"User"             {$FilterRegKey += "0x83","0x9c"}
			#"Operation"        {$FilterRegKey += "0x77","0x9c"}
			#"ImagePath"        {$FilterRegKey += "0x84","0x9c"}
			#"Session"          {$FilterRegKey += "0x85","0x9c"}
			#"Path"             {$FilterRegKey += "0x87","0x9c"}
			#"TID"              {$FilterRegKey += "0x88","0x9c"}
			#"Duration"         {$FilterRegKey += "0x8D","0x9c"}
			#"TimeOfDay"        {$FilterRegKey += "0x8E","0x9c"}
			#"Version"          {$FilterRegKey += "0x91","0x9c"}
			#"EventClass"       {$FilterRegKey += "0x92","0x9c"}
			#"AuthenticationID" {$FilterRegKey += "0x93","0x9c"}
			#"Virtualized"      {$FilterRegKey += "0x94","0x9c"}
			#"Integrity"        {$FilterRegKey += "0x95","0x9c"}
			#"Category"         {$FilterRegKey += "0x96","0x9c"}
			#"Parent PID"       {$FilterRegKey += "0x97","0x9c"}
			#"Architecture"     {$FilterRegKey += "0x98","0x9c"}
	      #  "Sequence"         {$FilterRegKey += "0x7A","0x9c"}	
			#"Company"          {$FilterRegKey += "0x80","0x9c"}
			#"Description"      {$FilterRegkey += "0x81","0x9c"}
			#default            {
			#					[string]$FilterRegKey = "Error: Check Column values."
			#				    return
			#				   }
		  # }
            $FilterRegKey += "0x84","0x9c" # davam natvrdo, lebo switch pridal hodnoty dvakrat
		   
		   #Add two zero bytes padding before comparison
		   $FilterRegkey += "0x0","0x0"
	       #Now add Relation byte
		   switch($_.Relation)
	       {
	        "is"         {$FilterRegKey += "0x0"}
	        "isNot"      {$FilterRegkey += "0x1"}
	        "lessThan"   {$filterregkey += "0x2"}
	        "moreThan"   {$FilterRegkey += "0x3"}
	        "endsWith"   {$FilterRegkey += "0x5"}
	        "BeginsWith" {$FilterRegkey += "0x4"}
		    "Contains"   {$FilterRegKEy += "0x6"}
	        "excludes"   {$FilterRegkey += "0x7"}
			default      {
							[string]$FilterRegKey = "Error: Check Relation values."
							return}
	       }
		    
		   #Add three zero bytes before Action (Include/Exclude)
		   $FilterRegKey += "0x0","0x0","0x0"
		   #Now Include/Exclude
	       if  ($_.Action -match "incl"){$FilterRegkey += "0x1"}
	       elseif($_.Action -match "excl"){$FilterRegKey += "0x0"}
		   else{[string]$FilterRegkey = "Error: Check Action Values.";return}
	       #Add length of <Value> string.
		   #Length is hex value of (characters * 2(account for nulls) + 2)(account for spacer bytes)
		   $NumPathChars = [Convert]::tostring(((($_.value.toCharArray() | measure).count *  2) + 2),"16")
		   $FilterRegKey += (convert-LargeValues -value $NumPathChars)
		   #Two zero bytes padding
		   $FilterRegkey += "0x0","0x0"
		   #Convert string "Value" to binary Ascii array (ie. A = 0x41)
		   $_.Value.toCharArray() | % {
	       	 $FilterRegkey += (convert-largeValues -value ([Convert]::ToString(([char]$_ -as [int]),"16")))
		   	}
	       #Current Filter calculated, pad with 10 zero bytes
		   $FilterRegkey += "0x0","0x0","0x0","0x0","0x0","0x0","0x0","0x0","0x0","0x0"
	     }                      
	     #Check for syntax errors
		 if($FilterRegkey -match "Error")
		 	{return ($FilterRegkey | sort | get-unique)}
			
		 #Set filter
		 New-ItemProperty "HKCU:\Software\Sysinternals\Process Monitor" "FilterRules" -Value $FilterRegKey `
		   -PropertyType Binary -Force -ErrorVariable SetRegKeyErr | Out-Null
		 if(($setRegKeyErr | measure).count -ne 0)
			 {Return "Error: Writing registry failed."}
		 else{return 0}
	 }

$ProcMon = "C:\Users\BPD\Desktop\Procmon.exe"
$HandleExe = "C:\Users\BPD\Desktop\handle.exe"
$ProcMonTestDir = "C:\Users\BPD\Desktop\dataset2"

Get-ChildItem $ProcMonTestDir | 
Foreach-Object {
write $_.BaseName
$FileLocked = $false

# make sure backing file isn't present in case it wasn't deleted on last run
$ProcMonBack = "C:\Users\BPD\Desktop\outputs\" + $_.BaseName + ".pml"

$FileExists = Test-Path $ProcMonBack
if ($FileExists -eq $true){
     do {
        Start-Sleep -seconds 1
        $TestFileLock = & $HandleExe $ProcMonBack /AcceptEula
        if ($TestFileLock.count -gt 0){
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
          }  else {
            $FileLocked = $false
            }
    } while( $FileLocked -eq $true )
   Remove-Item $ProcMonBack -force
}

$ProcMonTest = $_.FullName
$CSVFile = "C:\Users\BPD\Desktop\procout\"+ $_.BaseName + ".csv"

# filter
$Filter = "ImagePath,isNot," + $ProcMonTest +",Exclude"
$FilterObj = convert-FiltertoObj 
$FilterWriteResult = write-ProcmonFilterValue
 
& $ProcMon /Quiet /AcceptEula /Minimized /Backingfile $ProcMonBack

do {
    Start-Sleep -seconds 1 # procmon.exe /waitforidle doesn't appear to work well when scripted with PowerShell
    $ProcMonProcess = Get-Process | where {$_.Path -eq $ProcMon}
} while ( $ProcMonProcess.Id -eq $null )

& $ProcMonTest -WindowStyle Hidden
Start-Sleep -seconds 10  # cas ktory sa zaznamenava proces !!!!

$ProcMonTestProcess = Get-Process | where {$_.Path -eq $ProcMonTest} -ErrorAction SilentlyContinue
if ($ProcMonTestProcess) {
  # try gracefully first
  $ProcMonTestProcess.CloseMainWindow()
  # kill after five seconds
  Sleep 5
  if (!$ProcMonTestProcess.HasExited) {
    $ProcMonTestProcess | Stop-Process -Force
  }
}

#Stop-Process $ProcMonTestProcess.Id
#Wait-Process $ProcMonTestProcess.Id

#$ProcMonProcess = Get-Process | where {$_.Path -eq $ProcMon}

& $ProcMon /Terminate
#Start-Process -FilePath $ProcMon -argument "/terminate /accepteula" -Wait -WindowStyle Hidden

# Test for file lock on procmon.exe backing file before exporting
do {
    Start-Sleep -seconds 1
    $TestFileLock = & $HandleExe $ProcMonBack /AcceptEula
    if ($TestFileLock.count -gt 0){
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
      }  else {
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
        if ($TestFileLock.count -gt 0){
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }
            else{
                $FileLocked = $false
            }
          }  else {
            $FileLocked = $false
            }
    } while( $FileLocked -eq $true )
   Remove-Item $ProcMonBack -force
}

#cakam kym sa zapise do csv
$FileExists = Test-Path $CSVFile
if ($FileExists -eq $true){
do {
    Start-Sleep -seconds 1
    $TestFileLock = & $HandleExe $CSVFile /AcceptEula
    if ($TestFileLock.count -gt 0){
            if ($TestFileLock[-2] -match "pid:"){
                $FileLocked = $true
            }else{
                $FileLocked = $false
            }
  }  else {
            $FileLocked = $false
            }        
} while( $FileLocked -eq $true )
}
}