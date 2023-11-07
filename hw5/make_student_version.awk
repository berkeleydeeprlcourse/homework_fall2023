BEGIN {
    in_todo = 0;
    multi_line_comment = 0;
    comment_content = "";
    todo_indent = "";
}

# Detect the start of a TODO block
/^([ \t]*# TODO\(student\).*)$/ {
    in_todo = 1;
    print $0;
    next;
}

# If inside a TODO block, check for the start of a multi-line comment
in_todo && /^[ \t]*"""/ {
    multi_line_comment = !multi_line_comment;
    next;
}

# If inside a TODO block and a multi-line comment, capture the next line
in_todo && multi_line_comment && !/^[ \t]*"""/ && !/^[ \t]*# ENDTODO/ {
    comment_content = $0;
    print comment_content;  # Print the content inside multi-line comment
    next;
}

# If reaching the end of a TODO block, reset flags
/^[ \t]*# ENDTODO/ {
    in_todo = 0;
    multi_line_comment = 0;
    comment_content = "";
    todo_indent = "";
    next;
}

# If not inside a TODO block, print the line normally
!in_todo {
    print;
}
