#!/bin/zsh
# kn_run.sh NAME: open NAME.pptx in Keynote (activate in the same call, or the open event stalls), list fonts, export NAME_kn.pdf, close
D=$(cd "$(dirname "$0")" && pwd)   # the folder that holds NAME.pptx
N=$1
osascript -e 'with timeout of 300 seconds' -e 'tell application id "com.apple.Keynote"' -e 'activate' -e 'delay 2' -e "open (POSIX file \"$D/$N.pptx\")" -e 'delay 8' -e 'get name of documents' -e 'end tell' -e 'end timeout' 2>&1
osascript $D/kn_test.applescript "$D/$N.pptx" "$D/${N}_kn.pdf" "$N" > $D/kn_$N.txt 2>&1
head -n 2 $D/kn_$N.txt
grep "font=" $D/kn_$N.txt | grep -v "HiraginoSans-W6\|Calibri\|Times-Roman" | head -n 10
grep -c "Times-Roman" $D/kn_$N.txt
ls -la $D/${N}_kn.pdf
