-- kn_test.applescript <pptx> <pdf> <docname>: open a pptx in Keynote, list the font and size of every text item, export a PDF, close
on run argv
	set inPath to item 1 of argv
	set outPath to item 2 of argv
	set dn to item 3 of argv
	set res to ""
	with timeout of 600 seconds
		tell application id "com.apple.Keynote"
			repeat 6 times
				if (name of documents) contains dn then exit repeat
				try
					open (POSIX file inPath)
				end try
				delay 5
			end repeat
			set d to document dn
			set res to res & "name=" & (name of d) & " size=" & (width of d) & "x" & (height of d) & " slides=" & (count of slides of d) & linefeed
			repeat with i from 1 to (count of slides of d)
				set s to slide i of d
				set res to res & "=== S" & i & " text=" & (count of text items of s) & " images=" & (count of images of s) & " lines=" & (count of lines of s) & linefeed
				repeat with k from 1 to (count of text items of s)
					set ti to text item k of s
					try
						set p to position of ti
						set t to (object text of ti) as text
						if (count of characters of t) > 36 then set t to (text 1 thru 36 of t)
						set res to res & "  @(" & (item 1 of p) & "," & (item 2 of p) & ") " & (width of ti) & "x" & (height of ti) & " font=" & (font of object text of ti) & " size=" & (size of object text of ti) & " [" & t & "]" & linefeed
					on error e
						set res to res & "  ERR " & e & linefeed
					end try
				end repeat
			end repeat
			export d to (POSIX file outPath) as PDF
			close d saving no
		end tell
	end timeout
	return res
end run
