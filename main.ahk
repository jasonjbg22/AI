#SingleInstance, Force
#Persistent
#NoEnv
SendMode Input
SetWorkingDir, %A_ScriptDir%

global RootFolder := "G:\AllPhotosDatabase\2em"
global VideoRootFolder := "G:\AllPhotosDatabase\2em\Videos"
global ImageExtensions := "jpg,jpeg,png,gif,bmp,webp"
global VideoExtensions := "mp4,avi,mkv,mov,wmv,flv"

global AllFileList := []
global VideoFileList := []
global CurrentListRef
global CurrentPID := 0

all_ext := ImageExtensions . "," . VideoExtensions

Loop, Files, %RootFolder%\*.*, R
{
    if A_LoopFileExt in %all_ext%
    {
        AllFileList.Push(A_LoopFileFullPath)
    }
}

Loop, Files, %VideoRootFolder%\*.*, R
{
    if A_LoopFileExt in %VideoExtensions%
    {
        VideoFileList.Push(A_LoopFileFullPath)
    }
}
return

#b::
    if (AllFileList.Length() = 0)
    {
        MsgBox, 48, Error, No image or video files were found in: %RootFolder%
        return
    }
    CurrentListRef := AllFileList
    GoSub, ShowRandomMedia
    SetTimer, ShowRandomMedia, 5000
return

#v::
    if (VideoFileList.Length() = 0)
    {
        MsgBox, 48, Error, No video files were found in: %VideoRootFolder%
        return
    }
    CurrentListRef := VideoFileList
    GoSub, ShowRandomMedia
    SetTimer, ShowRandomMedia, 5000
return

#s::
    SetTimer, ShowRandomMedia, Off
    TrayTip, Slideshow Timer, Paused!, 10, 1
    Sleep, 2000
    HideTrayTip()
return

#Space::
    if (CurrentPID) 
    {
        GoSub, PauseSlideshow
    }
return

F1::SEND, {Left}
F2::SEND, {Right}

::em!::
    Run, D:\DataBackup\Ems\RadPics.ps1
return

::em!2::
    Loop, 7
    {
        Run, D:\DataBackup\Ems\RadPics.ps1
        sleep 100
    }
return

::em@::
    Run, D:\DataBackup\Ems\RadVids.ps1
return

ShowRandomMedia:
    if (CurrentPID)
    {
        WinClose, ahk_pid %CurrentPID%
        WinWaitClose, ahk_pid %CurrentPID%,, 1
        if (Process, Exist, CurrentPID)
        {
            Process, Close, %CurrentPID%
        }
        CurrentPID := 0
    }
    Random, randomIndex, 1, % CurrentListRef.Length()
    SelectedFile := CurrentListRef[randomIndex]
    
    Run, %SelectedFile%
    Sleep, 1200
    WinGet, CurrentPID, PID, A
return

PauseSlideshow:
    SetTimer, ShowRandomMedia, Off
    if (CurrentPID)
    {
        WinClose, ahk_pid %CurrentPID%
        WinWaitClose, ahk_pid %CurrentPID%,, 1
        if (Process, Exist, CurrentPID)
        {
            Process, Close, %CurrentPID%
        }
        CurrentPID := 0
    }
return

HideTrayTip() {
    TrayTip
    if SubStr(A_OSVersion,1,3) = "10." {
        Menu Tray, NoIcon
        Sleep 200
        Menu Tray, Icon
    }
}
